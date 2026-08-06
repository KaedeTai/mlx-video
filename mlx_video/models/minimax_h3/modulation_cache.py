"""Precomputed AdaLN modulation for MiniMax-H3 (mlx-video port).

This is a straight port of ``PipeNetwork/minimax-h3-mlx/minimax_h3_mlx/adaln.py`` adapted to
mlx-video's DiT layout. See that file for the original rationale.

Motivation
----------
~13B of MiniMax-H3's 33B parameters live in the per-block ``adaln_proj.linear`` projection
(50 blocks x ``[96768, 2688]``). Its input is the timestep embedding alone, which does not
depend on the packed sequence, so for a fixed sampler schedule every modulation tensor the
whole denoising run will ever need can be computed once and the projections then dropped.

In mlx-video's HEAVY_SUFFIX Q4 build the ``adaln_proj.linear`` weights are deliberately kept
in bf16 (they are excluded from ``HEAVY_SUFFIX`` in ``scripts/h3/quantize_dit.py``). That means
each 50-block adaln bank is ~26 GB in RAM. Precomputing the modulation from those bf16
projections into a ~387 MB lookup table (for a 4-step schedule) then dropping the weights
yields the same ~67x reduction the pipenetwork build reports while preserving voice quality.

v16 260807 change (P0 safety) — exact step-indexed cache
--------------------------------------------------------
Prior versions keyed the per-step modulation on the rounded floating-point ``unique_t`` tuple.
That relied on the model computing exactly the same float sequence at inference as at build.
Empirically it held for Turbo 4-step but **missed** at 30/50/100 steps (2 / 2 / 8 misses on
30 / 50 / 100 steps respectively) because ``time_shift_sigma`` compounded fp32 vs fp64 order
subtleties. Result: silent fallback to a slower per-step recompute if the ``adaln_proj``
weights were still around — or a hard KeyError if they had been dropped.

v16 fixes this by keying the cache on the **integer step index** the sampler drives. The
pipeline pokes ``payload["step_index"]`` before every DiT call, the DiT reads it, and the
cache's ``gather(block_idx, step_index)`` is now a pure ``list[block_idx][step_index]``
lookup with no float math. The ``per_step_ut`` list is still stored (for verification /
bitwise-parity checks) but never used as a hash key at runtime.

The cache also carries a full :class:`CacheSignature`. ``drop_adaln_weights`` refuses to
delete the projections unless the signature agrees with the live pipeline (num_steps,
schedule name, shift_video / shift_audio, curves-disabled, dtype, model/LoRA/layout hashes).
Any mismatch → refuse the drop and keep the ``adaln_proj`` weights so the model still
functions via its legacy path.

Note on pipenetwork adaln quantization
---------------------------------------
``~/models/MiniMax-H3-4bit/quant_config.json`` sets ``adaln_bits=8`` (with ``quantize_adaln:true``),
so pipenetwork keeps the adaln projections at **Q8**, not Q4. mlx-video's HEAVY_SUFFIX Q4 build
leaves adaln at **bf16** for maximum precision; the cache is stored in **fp32** by default so
repeated downcast noise (bf16 → fp32 → bf16) does not accumulate across 4-step denoising.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx


# The modality axis (video, text, audio) is baked into ``AdalnProj`` as ``modalities=3``
# in mlx-video (see ``blocks.py``). Kept as a constant here so future refactors of the
# modality count show up in one place.
MODALITY_NUM = 3

# Reference conditioning noise-levels used by ``MiniMaxH3Model.__call__`` to pick the
# per-conditioning-segment noise-shift. Mirror ``packed_layout.py`` verbatim.
VISUAL_COND_TIMESTEP = 0.999
AUDIO_COND_TIMESTEP = 1.0

# Match scheduler.set_timesteps: sigma_v is clamped to at least sigma_min to avoid
# divide-by-zero in the model's velocity math. Anything ``t_v = 1 - sigma_v`` computed
# in the model would follow the same clamp implicitly; we mirror it here so the cache
# union is exact.
_SIGMA_MIN_DEFAULT = 1e-5

# Schedule label baked into the signature. Any pipeline that swaps schedulers (e.g. from
# flow-matching Euler to Karras DPM++) must bump this so a stale cache is rejected.
DEFAULT_SCHEDULE_NAME = "flow_matching_default"

# ``drop_adaln_weights`` only fires if ``signature.num_steps`` is one of these. Anything
# else means the caller is running an experimental sampler and we refuse the drop.
DEFAULT_SUPPORTED_STEP_COUNTS: frozenset = frozenset({4, 8, 15, 20, 25, 30, 40, 50, 75, 100})


def _time_shift_sigma(sigma: float, fr: float, to: float) -> float:
    base = sigma / (fr + sigma * (1.0 - fr))
    return to * base / (1.0 + (to - 1.0) * base)


def _round_key(t: float) -> float:
    """Legacy float-key round (kept for backwards-compat with old signatures).

    v16: no longer used at runtime for the primary cache lookup — kept only for
    :func:`schedule_timesteps_with_keys` output stability and for computing a
    fingerprint of the unique_t list inside :class:`CacheSignature`.
    """
    return round(float(t), 9)


# ---------------------------------------------------------------------------
# CacheSignature — fail-closed identity of a built cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheSignature:
    """Everything about a run that must match before a cached modulation is trusted.

    Any field mismatch between the cache and the live pipeline → refuse to use the
    cache (and refuse to drop the ``adaln_proj`` weights). Serialisable to JSON so
    the stripped-bundle build can persist it alongside the cache NPZ.
    """

    num_steps: int
    num_blocks: int
    has_visual_cond: bool
    has_audio_cond: bool
    shift_video: float
    shift_audio: float
    sigma_min: float
    schedule_name: str
    dtype: str  # "float32" (default) — repeat cast to bf16 was banned in v15
    use_adaln_curves: bool  # must be False for the cache to be valid
    cache_final: bool
    # Content-addressed hashes — stable across runs on the same disk state.
    model_hash: str  # sha256 of the safetensors weights (or "unknown")
    lora_hash: Optional[str]  # sha256 of the LoRA weights + alpha, or None
    layout_hash: str  # hash of layout dims + ref shapes for the run
    per_step_ut_hash: str  # sha256 of the per-step unique_t list
    # Freeform tag so a caller can identify a bundle without diffing hashes.
    tag: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CacheSignature":
        # Filter unknown keys (forward-compat with older signatures) — but any
        # required key that is missing will still raise a TypeError.
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in allowed})

    def matches(self, other: "CacheSignature", *, ignore: Iterable[str] = ()) -> Tuple[bool, List[str]]:
        """Compare two signatures. Returns ``(ok, mismatched_fields)``."""
        skip = set(ignore)
        bad: List[str] = []
        for f in self.__dataclass_fields__:  # type: ignore[attr-defined]
            if f in skip:
                continue
            if getattr(self, f) != getattr(other, f):
                bad.append(f)
        return not bad, bad


def _sha256_hex(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


def _hash_per_step_ut(per_step_ut: Sequence[Sequence[float]]) -> str:
    """Deterministic hash of a list-of-lists of floats (rounded to 9dp)."""
    canon = [[round(float(t), 9) for t in step] for step in per_step_ut]
    return _sha256_hex(json.dumps(canon, sort_keys=False).encode("utf-8"))


def hash_layout(layout: Any) -> str:
    """Hash the fields of a :class:`PackedLayout` that affect adaln shape."""
    payload: Dict[str, Any] = {}
    for name in ("signature", "seq_len", "num_video", "num_audio", "num_text",
                 "num_cond_video", "num_cond_audio"):
        v = getattr(layout, name, None)
        if v is None:
            continue
        try:
            payload[name] = tuple(v) if hasattr(v, "__iter__") and not isinstance(v, str) else v
        except TypeError:
            payload[name] = repr(v)
    segs = getattr(layout, "segments", None)
    if segs is not None:
        payload["segments"] = [tuple(s) for s in segs]
    return _sha256_hex(json.dumps(payload, default=str, sort_keys=True).encode("utf-8"))


def hash_file(path: Any, *, chunk_bytes: int = 8 * 1024 * 1024,
              max_bytes: Optional[int] = None) -> str:
    """SHA-256 hex of a file. ``max_bytes`` limits how much we read (None = full file).

    For very large safetensors on a spinning disk pass ``max_bytes=64 * 1024**2`` to
    take a fast fingerprint of the leading 64 MB instead of the full 25 GiB.
    """
    from pathlib import Path
    p = Path(path).expanduser()
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    read = 0
    with p.open("rb") as fh:
        while True:
            n = chunk_bytes if max_bytes is None else min(chunk_bytes, max_bytes - read)
            if n <= 0:
                break
            buf = fh.read(n)
            if not buf:
                break
            h.update(buf)
            read += len(buf)
    h.update(str(p.stat().st_size).encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Schedule helpers (kept for build_stripped_bundle + backwards-compat callers)
# ---------------------------------------------------------------------------


def schedule_timesteps(
    sigmas: Sequence[float],
    *,
    has_visual_cond: bool = False,
    has_audio_cond: bool = False,
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    sigma_min: float = _SIGMA_MIN_DEFAULT,
    visual_cond_t: float = VISUAL_COND_TIMESTEP,
    audio_cond_t: float = AUDIO_COND_TIMESTEP,
) -> mx.array:
    """Build the union of distinct ``unique_t`` values a full denoising run will produce.

    Only used by legacy callers / diagnostics — the v16 gather path is step-index-keyed
    and never round-trips through this union.
    """
    ts: set = set()
    for sigma in list(sigmas)[:-1]:
        sigma_v = max(float(sigma), float(sigma_min))
        t_v = 1.0 - sigma_v
        t_a = 1.0 - _time_shift_sigma(sigma_v, float(shift_video), float(shift_audio))
        ts.add(_round_key(t_v))
        ts.add(_round_key(t_a))
        if has_visual_cond:
            ts.add(_round_key(max(t_v, float(visual_cond_t))))
        if has_audio_cond:
            ts.add(_round_key(max(t_a, float(audio_cond_t))))
    ordered = sorted(ts)
    return mx.array(ordered, dtype=mx.float32)


def schedule_timesteps_with_keys(*args, **kwargs) -> "tuple[list, mx.array]":
    """Companion to :func:`schedule_timesteps` that also returns the exact Python-float keys."""
    arr = schedule_timesteps(*args, **kwargs)
    ts_set: set = set()
    sigmas = list(args[0] if args else kwargs["sigmas"])[:-1]
    shift_v = kwargs.get("shift_video", 12.0)
    shift_a = kwargs.get("shift_audio", 3.0)
    sigma_min = kwargs.get("sigma_min", _SIGMA_MIN_DEFAULT)
    has_v = kwargs.get("has_visual_cond", False)
    has_a = kwargs.get("has_audio_cond", False)
    vt = kwargs.get("visual_cond_t", VISUAL_COND_TIMESTEP)
    at = kwargs.get("audio_cond_t", AUDIO_COND_TIMESTEP)
    for sigma in sigmas:
        sigma_v = max(float(sigma), float(sigma_min))
        t_v = 1.0 - sigma_v
        t_a = 1.0 - _time_shift_sigma(sigma_v, float(shift_v), float(shift_a))
        ts_set.add(_round_key(t_v))
        ts_set.add(_round_key(t_a))
        if has_v:
            ts_set.add(_round_key(max(t_v, float(vt))))
        if has_a:
            ts_set.add(_round_key(max(t_a, float(at))))
    return sorted(ts_set), arr


def per_step_unique_t(
    sigmas: Sequence[float],
    *,
    has_visual_cond: bool = False,
    has_audio_cond: bool = False,
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    sigma_min: float = _SIGMA_MIN_DEFAULT,
    visual_cond_t: float = VISUAL_COND_TIMESTEP,
    audio_cond_t: float = AUDIO_COND_TIMESTEP,
) -> List[List[float]]:
    """Mirror ``MiniMaxH3Model.__call__``'s per-step ``sorted(distinct)`` unique_t list.

    Returns one list per denoising step (excluding the terminal ``sigma_min``
    schedule entry that the model never receives).
    """
    per_step: List[List[float]] = []
    for sigma in list(sigmas)[:-1]:
        sigma_v = max(float(sigma), float(sigma_min))
        t_v = 1.0 - sigma_v
        t_a = 1.0 - _time_shift_sigma(sigma_v, float(shift_video), float(shift_audio))
        distinct = {_round_key(t_v), _round_key(t_a)}
        if has_visual_cond:
            distinct.add(_round_key(max(t_v, float(visual_cond_t))))
        if has_audio_cond:
            distinct.add(_round_key(max(t_a, float(audio_cond_t))))
        per_step.append(sorted(distinct))
    return per_step


# ---------------------------------------------------------------------------
# ModulationCache — v16 step-indexed
# ---------------------------------------------------------------------------


class ModulationCache:
    """Per-block AdaLN modulation, precomputed for a fixed integer step schedule.

    Layout: ``per_step_tables[step_idx][block_idx]`` is a 6-tuple of
    ``[M*3, hidden]`` arrays (M = len(per_step_ut[step_idx])).
    ``per_step_final[step_idx]`` is a 2-tuple of ``[M, hidden]`` arrays.

    All runtime lookups are ``list[i][j]``. There is no float-hash indirection.
    """

    def __init__(
        self,
        per_step_tables: List[List[Tuple[mx.array, ...]]],
        per_step_ut: List[List[float]],
        signature: CacheSignature,
        per_step_final: Optional[List[Tuple[mx.array, ...]]] = None,
    ):
        if not per_step_tables:
            raise ValueError("per_step_tables must be non-empty")
        self.per_step_tables = per_step_tables
        self.per_step_ut = per_step_ut
        self.per_step_final = per_step_final
        self.signature = signature

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def num_steps(self) -> int:
        return len(self.per_step_tables)

    @property
    def num_blocks(self) -> int:
        return len(self.per_step_tables[0]) if self.per_step_tables else 0

    @property
    def num_timesteps(self) -> int:
        """Union size (for diagnostics only — the cache is step-indexed)."""
        seen = set()
        for step in self.per_step_ut:
            for t in step:
                seen.add(_round_key(t))
        return len(seen)

    def nbytes(self) -> int:
        total = 0
        for step in self.per_step_tables:
            for tup in step:
                total += sum(a.nbytes for a in tup)
        if self.per_step_final is not None:
            for tup in self.per_step_final:
                total += sum(a.nbytes for a in tup)
        return total

    def unique_t_for_step(self, step_index: int) -> List[float]:
        """Return the ``unique_t`` list this cache was built against for ``step_index``."""
        if step_index < 0 or step_index >= self.num_steps:
            raise IndexError(f"step_index {step_index} out of range [0, {self.num_steps})")
        return list(self.per_step_ut[step_index])

    # ------------------------------------------------------------------
    # Step-indexed gather (v16)
    # ------------------------------------------------------------------

    def gather(self, block_idx: int, step_index: int) -> Tuple[mx.array, ...]:
        """Return the 6-tuple of modulation tensors for ``(block_idx, step_index)``.

        Pure ``list[step_idx][block_idx]`` lookup. Raises ``IndexError`` on OOB.
        """
        if step_index < 0 or step_index >= self.num_steps:
            raise IndexError(
                f"ModulationCache miss: step_index={step_index} not in [0, {self.num_steps})"
            )
        if block_idx < 0 or block_idx >= self.num_blocks:
            raise IndexError(
                f"ModulationCache miss: block_idx={block_idx} not in [0, {self.num_blocks})"
            )
        return self.per_step_tables[step_index][block_idx]

    def final_layer_gather(self, step_index: int) -> Tuple[mx.array, mx.array]:
        """Return the 2-tuple ``(shift, scale)`` for ``step_index`` at the final layer."""
        if self.per_step_final is None:
            raise RuntimeError(
                "ModulationCache has no final_layer table; rebuild with cache_final=True "
                "or fall back to final_layer.adaln_proj(t_emb)."
            )
        if step_index < 0 or step_index >= self.num_steps:
            raise IndexError(
                f"ModulationCache final-layer miss: step_index={step_index} not in "
                f"[0, {self.num_steps})"
            )
        return self.per_step_final[step_index]

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        dit,
        per_step_ut: Sequence[Sequence[float]],
        signature: CacheSignature,
        *,
        dtype: mx.Dtype = mx.float32,
        cache_final: bool = True,
    ) -> "ModulationCache":
        """Precompute one modulation entry per (step, block) using live M-batching.

        ``per_step_ut`` mirrors what ``MiniMaxH3Model.__call__`` computes each step;
        it is the source of truth for both the build and the runtime gather. Building
        with the same M-batching as the live forward guarantees the cache is a
        bitwise drop-in replacement (v15 260807 bugfix #1).

        The ``signature`` is stored verbatim and compared against the live pipeline
        before ``drop_adaln_weights`` will fire (see :func:`drop_adaln_weights`).
        """
        if not per_step_ut:
            raise ValueError("per_step_ut must be non-empty")

        per_step_tables: List[List[Tuple[mx.array, ...]]] = []
        per_step_final: Optional[List[Tuple[mx.array, ...]]] = [] if cache_final else None

        for step_idx, ut in enumerate(per_step_ut):
            t_vals = mx.array([float(t) for t in ut], dtype=mx.float32)
            t_emb = dit.time_embedder(t_vals)
            mx.eval(t_emb)
            block_outs: List[Tuple[mx.array, ...]] = []
            for block in dit.blocks:
                out = tuple(a.astype(dtype) for a in block.adaln_proj(t_emb))
                mx.eval(out)
                block_outs.append(out)
            per_step_tables.append(block_outs)
            if cache_final:
                out_f = tuple(a.astype(dtype) for a in dit.final_layer.adaln_proj(t_emb))
                mx.eval(out_f)
                per_step_final.append(out_f)  # type: ignore[union-attr]

        return cls(
            per_step_tables=per_step_tables,
            per_step_ut=[list(ut) for ut in per_step_ut],
            per_step_final=per_step_final,
            signature=signature,
        )


# ---------------------------------------------------------------------------
# Bitwise verify
# ---------------------------------------------------------------------------


def _has_nan_or_inf(a: mx.array) -> bool:
    return bool(mx.any(mx.isnan(a)).item() or mx.any(mx.isinf(a)).item())


def verify_cache_bitwise(
    cache: "ModulationCache",
    dit,
    *,
    verify_final: bool = True,
    verbose: bool = True,
) -> Tuple[int, float]:
    """Bitwise-verify a step-indexed cache against a live ``dit``.

    Iterates the cache's own ``per_step_ut`` (i.e. the schedule the cache was built
    for). Raises ``RuntimeError`` on the first mismatch. Returns
    ``(total_tuple_checks, max_diff_seen)`` on success — ``max_diff_seen`` is 0.0
    when everything matches.
    """
    # NaN/Inf sweep first — fails fast on catastrophic build.
    for si, step_blocks in enumerate(cache.per_step_tables):
        for bi, tup in enumerate(step_blocks):
            for ti, arr in enumerate(tup):
                if _has_nan_or_inf(arr):
                    raise RuntimeError(
                        f"[adaln-cache] NaN/Inf in step={si} block={bi} tuple[{ti}] "
                        f"shape={arr.shape}"
                    )
    if cache.per_step_final is not None:
        for si, tup in enumerate(cache.per_step_final):
            for ti, arr in enumerate(tup):
                if _has_nan_or_inf(arr):
                    raise RuntimeError(
                        f"[adaln-cache] NaN/Inf in final_layer step={si} tuple[{ti}] "
                        f"shape={arr.shape}"
                    )

    n_blocks = len(dit.blocks)
    if n_blocks != cache.num_blocks:
        raise RuntimeError(
            f"[adaln-cache] block-count mismatch: cache has {cache.num_blocks} "
            f"blocks, dit has {n_blocks}"
        )
    total_checks = 0
    max_diff_seen = 0.0

    for step_idx, unique_t in enumerate(cache.per_step_ut):
        M = len(unique_t)
        t_vals = mx.array([float(t) for t in unique_t], dtype=mx.float32)
        t_emb = dit.time_embedder(t_vals)
        mx.eval(t_emb)
        for bi, block in enumerate(dit.blocks):
            live = block.adaln_proj(t_emb)
            cached = cache.gather(bi, step_idx)
            for ti, (lv, cv) in enumerate(zip(live, cached)):
                if tuple(lv.shape) != tuple(cv.shape):
                    raise RuntimeError(
                        f"[adaln-cache] shape mismatch block={bi} step={step_idx} "
                        f"tuple={ti}: live={tuple(lv.shape)} cached={tuple(cv.shape)}"
                    )
                lv_f = lv.astype(mx.float32)
                cv_f = cv.astype(mx.float32)
                if bool(mx.array_equal(lv_f, cv_f).item()):
                    total_checks += 1
                    continue
                diff = float(mx.max(mx.abs(lv_f - cv_f)).item())
                if diff > max_diff_seen:
                    max_diff_seen = diff
                if diff != 0.0:
                    raise RuntimeError(
                        f"[adaln-cache] BITWISE MISMATCH block={bi} step={step_idx} "
                        f"tuple={ti} M={M} unique_t={unique_t}: "
                        f"max|live - cached|={diff:.6e} (must be 0.0)"
                    )
                total_checks += 1

    if verify_final and cache.per_step_final is not None:
        for step_idx, unique_t in enumerate(cache.per_step_ut):
            t_vals = mx.array([float(t) for t in unique_t], dtype=mx.float32)
            t_emb = dit.time_embedder(t_vals)
            mx.eval(t_emb)
            live = dit.final_layer.adaln_proj(t_emb)
            cached = cache.final_layer_gather(step_idx)
            for ti, (lv, cv) in enumerate(zip(live, cached)):
                if tuple(lv.shape) != tuple(cv.shape):
                    raise RuntimeError(
                        f"[adaln-cache] final_layer shape mismatch step={step_idx} "
                        f"tuple={ti}: live={tuple(lv.shape)} cached={tuple(cv.shape)}"
                    )
                lv_f = lv.astype(mx.float32)
                cv_f = cv.astype(mx.float32)
                if bool(mx.array_equal(lv_f, cv_f).item()):
                    total_checks += 1
                    continue
                diff = float(mx.max(mx.abs(lv_f - cv_f)).item())
                if diff != 0.0:
                    raise RuntimeError(
                        f"[adaln-cache] final_layer BITWISE MISMATCH step={step_idx} "
                        f"tuple={ti}: max|live - cached|={diff:.6e} (must be 0.0)"
                    )
                total_checks += 1

    if verbose:
        final_str = " + final_layer" if (verify_final and cache.per_step_final is not None) else ""
        print(f"[adaln-cache] BITWISE VERIFY OK: {total_checks} tuple comparisons "
              f"({cache.num_blocks} blocks x {cache.num_steps} steps x 6 tuples{final_str}) "
              f"all mx.array_equal, no NaN/Inf")

    return total_checks, max_diff_seen


# ---------------------------------------------------------------------------
# Weight-drop helper — fail-closed on signature mismatch
# ---------------------------------------------------------------------------


class DropRefused(RuntimeError):
    """Raised when ``drop_adaln_weights`` refuses to delete projections.

    The caller should catch this and either (a) fall back to the legacy adaln path
    (leave the ``adaln_proj`` weights loaded) or (b) fail the run explicitly.
    """


def _live_signature_from_pipeline(
    *,
    num_steps: int,
    num_blocks: int,
    has_visual_cond: bool,
    has_audio_cond: bool,
    shift_video: float,
    shift_audio: float,
    sigma_min: float,
    schedule_name: str,
    dtype: str,
    use_adaln_curves: bool,
    cache_final: bool,
    model_hash: str,
    lora_hash: Optional[str],
    layout_hash: str,
    per_step_ut: Sequence[Sequence[float]],
    tag: str = "",
) -> CacheSignature:
    return CacheSignature(
        num_steps=int(num_steps),
        num_blocks=int(num_blocks),
        has_visual_cond=bool(has_visual_cond),
        has_audio_cond=bool(has_audio_cond),
        shift_video=float(shift_video),
        shift_audio=float(shift_audio),
        sigma_min=float(sigma_min),
        schedule_name=str(schedule_name),
        dtype=str(dtype),
        use_adaln_curves=bool(use_adaln_curves),
        cache_final=bool(cache_final),
        model_hash=str(model_hash),
        lora_hash=lora_hash if lora_hash is None else str(lora_hash),
        layout_hash=str(layout_hash),
        per_step_ut_hash=_hash_per_step_ut(per_step_ut),
        tag=str(tag),
    )


def drop_adaln_weights(
    dit,
    cache: "ModulationCache",
    *,
    live_signature: Optional[CacheSignature] = None,
    drop_final: bool = True,
    supported_step_counts: Iterable[int] = DEFAULT_SUPPORTED_STEP_COUNTS,
    verify_diff: Optional[float] = 0.0,
    verbose: bool = True,
) -> int:
    """Delete the per-block ``adaln_proj.linear`` parameters. Fail-closed.

    Preconditions (any failure → :class:`DropRefused`):

    1. ``cache.signature.num_steps`` must be in ``supported_step_counts``.
    2. ``cache.signature.schedule_name`` must be :data:`DEFAULT_SCHEDULE_NAME`.
    3. ``cache.signature.use_adaln_curves`` must be ``False``.
    4. If ``live_signature`` is given, ``cache.signature`` must match it in every
       field except ``tag``.
    5. If ``verify_diff`` is given and > 0.0, the drop is refused (a nonzero
       live-vs-cache diff was seen during verification).

    Returns bytes freed on success. On failure raises :class:`DropRefused`
    without touching any weights.
    """
    sig = cache.signature
    reasons: List[str] = []

    if sig.num_steps not in set(int(x) for x in supported_step_counts):
        reasons.append(
            f"num_steps={sig.num_steps} not in supported set {sorted(supported_step_counts)}"
        )
    if sig.schedule_name != DEFAULT_SCHEDULE_NAME:
        reasons.append(
            f"schedule_name={sig.schedule_name!r} != {DEFAULT_SCHEDULE_NAME!r}"
        )
    if sig.use_adaln_curves:
        reasons.append("use_adaln_curves is True (cache assumes silu-then-linear path)")
    if verify_diff is not None and verify_diff > 0.0:
        reasons.append(f"verify_cache_bitwise max_diff={verify_diff:.6e} > 0")

    if live_signature is not None:
        ok, bad = sig.matches(live_signature, ignore=("tag",))
        if not ok:
            reasons.append(f"signature mismatch on fields: {bad}")

    if reasons:
        msg = "[adaln-cache] drop refused: " + "; ".join(reasons)
        if verbose:
            print(msg)
        raise DropRefused(msg)

    # ---- Actually drop ------------------------------------------------

    def _drop_arrays(m) -> int:
        b = 0
        for name in ("weight", "bias", "scales", "biases", "lora_A", "lora_B"):
            param = getattr(m, name, None)
            if isinstance(param, mx.array):
                b += param.nbytes
                delattr(m, name)
        base = getattr(m, "base", None)
        if base is not None and base is not m:
            b += _drop_arrays(base)
        return b

    freed = 0
    for block in dit.blocks:
        freed += _drop_arrays(block.adaln_proj.linear)
    if drop_final:
        freed += _drop_arrays(dit.final_layer.adaln_proj.linear)

    try:
        mx.clear_cache()
    except AttributeError:
        pass

    return freed


__all__ = [
    "MODALITY_NUM",
    "VISUAL_COND_TIMESTEP",
    "AUDIO_COND_TIMESTEP",
    "DEFAULT_SCHEDULE_NAME",
    "DEFAULT_SUPPORTED_STEP_COUNTS",
    "CacheSignature",
    "DropRefused",
    "ModulationCache",
    "drop_adaln_weights",
    "verify_cache_bitwise",
    "schedule_timesteps",
    "schedule_timesteps_with_keys",
    "per_step_unique_t",
    "hash_file",
    "hash_layout",
    "_live_signature_from_pipeline",
    "_hash_per_step_ut",
]
