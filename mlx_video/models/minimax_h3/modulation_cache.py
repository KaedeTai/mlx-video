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

Note on pipenetwork adaln quantization
---------------------------------------
``~/models/MiniMax-H3-4bit/quant_config.json`` sets ``adaln_bits=8`` (with ``quantize_adaln:true``),
so pipenetwork keeps the adaln projections at **Q8**, not Q4. mlx-video's HEAVY_SUFFIX Q4 build
leaves adaln at **bf16** for maximum precision; the cache is stored in **fp32** by default so
repeated downcast noise (bf16 → fp32 → bf16) does not accumulate across 4-step denoising.

mlx-video specifics
-------------------
Unlike pipenetwork's DiT, mlx-video's ``MiniMaxH3Model.__call__`` recomputes ``unique_t`` per
step (a subset of the full schedule union). Modulation for a step is therefore *gathered* from
the global cache by looking up each ``unique_t`` value's row in the global table.

Row layout of a cached block: ``[T_all * 3, hidden]`` — exactly the layout ``AdalnProj``
produces when fed ``t_emb`` of shape ``[T_all, time_embed_dim]``. Six such arrays per block
(shift/scale/gate for MSA and MLP). ``gather(block_idx, unique_t)`` slices the rows matching
``unique_t`` in that order, producing a ``[len(unique_t) * 3, hidden]`` tuple compatible with
``block.adaln_proj(t_emb)`` for the current step.

The final layer's ``adaln_proj`` (``expand=2, modalities=1``, ``[10752, 2688]``) is also
cached — smaller (~29M params) but avoids a redundant projection per step.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

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


def _time_shift_sigma(sigma: float, fr: float, to: float) -> float:
    base = sigma / (fr + sigma * (1.0 - fr))
    return to * base / (1.0 + (to - 1.0) * base)


def _round_key(t: float) -> float:
    """Round a timestep value to a deterministic 9-dp key for cache lookup.

    ``schedule_timesteps`` and ``gather`` both convert their inputs through this so
    a float that survives ``float32 -> float`` round-trips still lands on the same
    cache row.
    """
    return round(float(t), 9)


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

    Mirrors the ``unique_t`` derivation inside ``MiniMaxH3Model.__call__``:

    * for every scheduler sigma in the schedule (excluding the terminal ``sigma_min`` entry,
      which is only used as the tail of an Euler step, not fed to the model)::

          t_v = 1 - max(sigma, sigma_min)
          t_a = 1 - time_shift_sigma(max(sigma, sigma_min), shift_v, shift_a)

    * if visual conditioning rows are present the cond segment sits at ``max(t_v, 0.999)``;
    * if audio conditioning rows are present the ref-audio segment sits at ``max(t_a, 1.0)``.

    Returns
    -------
    mx.array : ``(T,)`` float32 sorted-ascending union of distinct unique-t values.
    """
    ts: set = set()
    # scheduler.sigmas has length ``N + 1`` (endpoints inclusive); only ``sigmas[:-1]`` is
    # ever fed to the model as ``timestep_for(i)``.
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
    """Companion to :func:`schedule_timesteps` that also returns the exact Python-float keys.

    Use this when passing the result to :meth:`ModulationCache.build` — the keys are then
    threaded into ``ModulationCache.__init__(key_values=...)`` so runtime ``gather()`` calls
    can look up unique-t values whose fp64 representation differs from the fp32-cast
    ``timesteps`` array's ``tolist()`` output.
    """
    arr = schedule_timesteps(*args, **kwargs)
    # Recompute the ordered key list in fp64 (matches what :func:`schedule_timesteps` sorted).
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


class ModulationCache:
    """Per-block AdaLN modulation, precomputed for a fixed union of timesteps.

    Parameters
    ----------
    tables : list of 6-tuples, one per DiT block, each entry shape ``[T*3, hidden]``.
    timesteps : sorted-ascending float32 ``(T,)`` — the union of unique-t values.
    final_table : optional 2-tuple for ``FinalLayer.adaln_proj``, each shape ``[T, hidden]``.
    """

    def __init__(
        self,
        tables: List[Tuple[mx.array, ...]],
        timesteps: mx.array,
        final_table: Optional[Tuple[mx.array, mx.array]] = None,
        key_values: Optional[Sequence[float]] = None,
        per_step_tables: Optional[List[List[Tuple[mx.array, ...]]]] = None,
        per_step_final: Optional[List[Tuple[mx.array, ...]]] = None,
        step_signatures: Optional[Sequence[Tuple[float, ...]]] = None,
    ):
        self.tables = tables
        self.timesteps = timesteps
        self.final_table = final_table
        # v15 260807 bugfix #1: optional per-step tables that were built with the
        # same M-batching the live forward uses. Preferred at gather time when
        # available; falls back to the union table for out-of-schedule queries.
        self.per_step_tables = per_step_tables
        self.per_step_final = per_step_final
        self._step_signature_to_idx: dict = {}
        if step_signatures is not None:
            for i, sig in enumerate(step_signatures):
                self._step_signature_to_idx[tuple(sig)] = i
        # Build key -> row index once. Prefer ``key_values`` (typically the exact
        # Python floats used by :func:`schedule_timesteps`) so runtime lookups
        # aren't defeated by float32 quantization of the ``timesteps`` array.
        if key_values is None:
            key_values = timesteps.tolist()
        vals = [_round_key(v) for v in key_values]
        self._key_to_row: dict = {v: i for i, v in enumerate(vals)}

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def num_blocks(self) -> int:
        return len(self.tables)

    @property
    def num_timesteps(self) -> int:
        return int(self.timesteps.shape[0])

    def nbytes(self) -> int:
        total = sum(t.nbytes for tbl in self.tables for t in tbl)
        if self.final_table is not None:
            total += sum(t.nbytes for t in self.final_table)
        if self.per_step_tables is not None:
            for block_steps in self.per_step_tables:
                for step_tuple in block_steps:
                    total += sum(a.nbytes for a in step_tuple)
        if self.per_step_final is not None:
            for step_tuple in self.per_step_final:
                total += sum(a.nbytes for a in step_tuple)
        return total

    # ------------------------------------------------------------------
    # Row-index gather (block modulation)
    # ------------------------------------------------------------------

    def _row_indices(self, unique_t: Iterable[float]) -> mx.array:
        """Global row indices for a per-step ``unique_t`` list.

        For each ``t`` in ``unique_t`` (in order) three consecutive rows are emitted
        (one per modality tag: video=0, text=1, audio=2). Order follows the caller's
        ``unique_t`` list, which by convention (``sorted(distinct)`` in model.py) is
        ascending; that matches the row order ``AdalnProj`` would have produced if it
        had been called with ``t_emb`` gathered from the same list.
        """
        rows: List[int] = []
        for t in unique_t:
            key = _round_key(t)
            base = self._key_to_row.get(key)
            if base is None:
                raise KeyError(
                    f"ModulationCache miss for t={key!r}; cached timesteps are "
                    f"{sorted(self._key_to_row)}"
                )
            base *= MODALITY_NUM
            for tag in range(MODALITY_NUM):
                rows.append(base + tag)
        return mx.array(rows, dtype=mx.int32)

    def gather(self, block_idx: int, unique_t: Sequence[float]) -> Tuple[mx.array, ...]:
        """Return a 6-tuple of ``[len(unique_t) * 3, hidden]`` arrays for a block.

        Drop-in replacement for ``block.adaln_proj(t_emb)`` where ``t_emb`` was built
        from ``unique_t``.

        Prefers the per-step table (built with matched M-batching) when the caller
        supplies a ``unique_t`` matching a known step signature. Falls back to the
        union table (row-gather) otherwise.
        """
        sig = tuple(_round_key(t) for t in unique_t)
        if self.per_step_tables is not None:
            step_idx = self._step_signature_to_idx.get(sig)
            if step_idx is not None:
                return self.per_step_tables[block_idx][step_idx]
        idx = self._row_indices(unique_t)
        return tuple(mx.take(a, idx, axis=0) for a in self.tables[block_idx])

    def final_layer_gather(self, unique_t: Sequence[float]) -> Tuple[mx.array, mx.array]:
        """Return a 2-tuple ``(shift, scale)`` of ``[len(unique_t), hidden]`` arrays.

        Drop-in replacement for ``model.final_layer.adaln_proj(t_emb)``. Prefers the
        per-step final table when available (v15 260807 bugfix #1 -- matches live
        M-batching bitwise); falls back to union-row-gather otherwise.
        """
        sig = tuple(_round_key(t) for t in unique_t)
        if self.per_step_final is not None:
            step_idx = self._step_signature_to_idx.get(sig)
            if step_idx is not None:
                return self.per_step_final[step_idx]
        if self.final_table is None:
            raise RuntimeError(
                "ModulationCache has no final_layer table; rebuild with cache_final=True "
                "or fall back to final_layer.adaln_proj(t_emb)."
            )
        rows: List[int] = []
        for t in unique_t:
            key = _round_key(t)
            base = self._key_to_row.get(key)
            if base is None:
                raise KeyError(
                    f"ModulationCache miss (final layer) for t={key!r}."
                )
            rows.append(base)
        idx = mx.array(rows, dtype=mx.int32)
        return tuple(mx.take(a, idx, axis=0) for a in self.final_table)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        dit,
        timesteps: mx.array,
        *,
        dtype: mx.Dtype = mx.float32,
        cache_final: bool = True,
        key_values: Optional[Sequence[float]] = None,
        per_step_ut: Optional[Sequence[Sequence[float]]] = None,
    ) -> "ModulationCache":
        """Precompute the modulation table for every DiT block from a running DiT.

        Args:
            dit: a ``MiniMaxH3Model`` whose ``adaln_proj`` weights are still loaded (bf16
                for a HEAVY_SUFFIX Q4 build).
            timesteps: ``(T,)`` float32 sorted-ascending union of distinct unique-t values.
            dtype: storage dtype (default fp32 -- do not downcast, see 260807 note).
            cache_final: also cache ``final_layer.adaln_proj`` (default True).
            per_step_ut: optional list of per-step unique_t lists. When provided, an
                additional per-step table is built using the SAME M-batching the live
                forward uses (see v15 260807 bugfix #1). Runtime :meth:`gather` prefers
                the per-step table when the caller's ``unique_t`` matches a known
                step signature; falls back to the union table otherwise.

        v15 260807 bugfix #1: MLX Metal GEMM produces bit-different results for
        ``[1, T_DIM] @ W`` vs ``[k, T_DIM] @ W`` (row i). Global union-based build
        (M=1 per row) therefore diverges from live (M=k per step) by ~1e-3 -- small
        per row but visibly compounding over 4 denoise steps. Building a per-step
        table with matched M fixes this bitwise.
        """
        T_all = int(timesteps.shape[0])

        # ---- Fallback global-union table (M=1) -- used only when the caller
        # queries a unique_t list that doesn't match any known step signature.
        tables: List[Tuple[mx.array, ...]] = []
        for block in dit.blocks:
            step_outs: List[Tuple[mx.array, ...]] = []
            for i in range(T_all):
                t_single = timesteps[i:i + 1]
                t_emb_single = dit.time_embedder(t_single)
                out = tuple(a.astype(dtype) for a in block.adaln_proj(t_emb_single))
                mx.eval(out)
                step_outs.append(out)
            table = tuple(
                mx.concatenate([step_outs[i][j] for i in range(T_all)], axis=0)
                for j in range(len(step_outs[0]))
            )
            mx.eval(table)
            tables.append(table)

        final_table: Optional[Tuple[mx.array, mx.array]] = None
        if cache_final:
            step_finals: List[Tuple[mx.array, ...]] = []
            for i in range(T_all):
                t_single = timesteps[i:i + 1]
                t_emb_single = dit.time_embedder(t_single)
                out = tuple(a.astype(dtype) for a in dit.final_layer.adaln_proj(t_emb_single))
                mx.eval(out)
                step_finals.append(out)
            final_table = tuple(
                mx.concatenate([step_finals[i][j] for i in range(T_all)], axis=0)
                for j in range(len(step_finals[0]))
            )
            mx.eval(final_table)

        # ---- Per-step table (matches live M-batching) --------------------
        per_step_tables: Optional[List[List[Tuple[mx.array, ...]]]] = None
        per_step_final: Optional[List[Tuple[mx.array, ...]]] = None
        step_signatures: List[Tuple[float, ...]] = []
        if per_step_ut is not None:
            per_step_tables = [[None] * len(per_step_ut) for _ in range(len(dit.blocks))]  # type: ignore
            step_signatures = [tuple(_round_key(t) for t in ut) for ut in per_step_ut]
            for step_idx, ut in enumerate(per_step_ut):
                t_vals = mx.array([float(t) for t in ut], dtype=mx.float32)
                t_emb = dit.time_embedder(t_vals)
                mx.eval(t_emb)
                for bi, block in enumerate(dit.blocks):
                    out = tuple(a.astype(dtype) for a in block.adaln_proj(t_emb))
                    mx.eval(out)
                    per_step_tables[bi][step_idx] = out
                if cache_final:
                    if per_step_final is None:
                        per_step_final = [None] * len(per_step_ut)  # type: ignore
                    out_f = tuple(a.astype(dtype) for a in dit.final_layer.adaln_proj(t_emb))
                    mx.eval(out_f)
                    per_step_final[step_idx] = out_f

        return cls(
            tables, timesteps, final_table=final_table, key_values=key_values,
            per_step_tables=per_step_tables,
            per_step_final=per_step_final,
            step_signatures=step_signatures if step_signatures else None,
        )


# ---------------------------------------------------------------------------
# Weight-drop helper
# ---------------------------------------------------------------------------


def drop_adaln_weights(dit, drop_final: bool = True) -> int:
    """Delete the per-block ``adaln_proj.linear`` parameters after a cache has been built.

    Returns the number of **bytes** freed. Only safe once a :class:`ModulationCache` covering
    the whole schedule exists — the block stack will KeyError if asked to modulate a timestep
    that's not in the cache.

    Handles three shapes ``adaln_proj.linear`` can take in the mlx-video build:

    * bare ``nn.Linear``           - drop ``weight`` / ``bias``.
    * bf16 or Q4 ``nn.Linear``     - Q4 also carries ``scales`` / ``biases``.
    * ``_LoRAOverlay`` wrapper     - drop ``lora_A`` / ``lora_B`` plus recurse into
      the wrapped ``.base`` module.
    """
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

    # Force MLX to actually release the Metal buffers now (arrays we just delattr'd
    # are unreachable, but the arena keeps them until cache is cleared).
    try:
        mx.clear_cache()
    except AttributeError:
        pass

    return freed

__all__ = [
    "MODALITY_NUM",
    "ModulationCache",
    "drop_adaln_weights",
    "schedule_timesteps",
    "schedule_timesteps_with_keys",
    "VISUAL_COND_TIMESTEP",
    "AUDIO_COND_TIMESTEP",
]


# ---------------------------------------------------------------------------
# v15 260807 bugfix #1/#2: rigorous per-step bitwise verification
# ---------------------------------------------------------------------------


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


def _has_nan_or_inf(a: mx.array) -> bool:
    return bool(mx.any(mx.isnan(a)).item() or mx.any(mx.isinf(a)).item())


def verify_cache_bitwise(
    cache: "ModulationCache",
    dit,
    per_step_ut: Sequence[Sequence[float]],
    *,
    verify_final: bool = True,
    verbose: bool = True,
) -> None:
    """Raise ``RuntimeError`` unless cached rows exactly match a live ``adaln_proj``
    call, for EVERY block, EVERY step, and (optionally) the final layer.

    Comparison is bit-exact (``mx.array_equal``) — the cache is built in fp32 and
    live is computed in fp32, so any nonzero difference indicates the cache is
    not a valid drop-in replacement.

    Also verifies no NaN/Inf in any cached table.
    """
    # ---- NaN/Inf sweep first (cheap; fails fast on catastrophic build) ----
    for bi, table in enumerate(cache.tables):
        for ti, arr in enumerate(table):
            if _has_nan_or_inf(arr):
                raise RuntimeError(
                    f"[adaln-cache] NaN/Inf in block {bi} tuple[{ti}] shape={arr.shape}"
                )
    if cache.final_table is not None:
        for ti, arr in enumerate(cache.final_table):
            if _has_nan_or_inf(arr):
                raise RuntimeError(
                    f"[adaln-cache] NaN/Inf in final_layer tuple[{ti}] shape={arr.shape}"
                )

    n_blocks = len(dit.blocks)
    n_steps = len(per_step_ut)
    total_checks = 0
    max_diff_seen = 0.0
    worst_loc: Optional[Tuple[int, int, int]] = None  # (block_idx, step_idx, tuple_idx)

    for step_idx, unique_t in enumerate(per_step_ut):
        M = len(unique_t)
        t_vals = mx.array(unique_t, dtype=mx.float32)
        t_emb = dit.time_embedder(t_vals)
        mx.eval(t_emb)
        for bi, block in enumerate(dit.blocks):
            live = block.adaln_proj(t_emb)  # 6-tuple of [M*3, hidden]
            cached = cache.gather(bi, unique_t)  # 6-tuple of [M*3, hidden]
            for ti, (lv, cv) in enumerate(zip(live, cached)):
                # Shape check first
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
                    worst_loc = (bi, step_idx, ti)
                if diff != 0.0:
                    raise RuntimeError(
                        f"[adaln-cache] BITWISE MISMATCH block={bi} step={step_idx} "
                        f"tuple={ti} M={M} unique_t={unique_t}: "
                        f"max|live - cached|={diff:.6e} (must be 0.0)"
                    )
                total_checks += 1

    if verify_final and cache.final_table is not None:
        for step_idx, unique_t in enumerate(per_step_ut):
            t_vals = mx.array(unique_t, dtype=mx.float32)
            t_emb = dit.time_embedder(t_vals)
            mx.eval(t_emb)
            live = dit.final_layer.adaln_proj(t_emb)  # 2-tuple of [M, hidden]
            cached = cache.final_layer_gather(unique_t)
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
        n_blocks_v = len(dit.blocks)
        final_str = " + final_layer" if (verify_final and cache.final_table is not None) else ""
        print(f"[adaln-cache] BITWISE VERIFY OK: {total_checks} tuple comparisons "
              f"({n_blocks_v} blocks x {n_steps} steps x 6 tuples{final_str}) "
              f"all mx.array_equal, no NaN/Inf")


__all__ += ["per_step_unique_t", "verify_cache_bitwise"]
