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
yields the same ~67x reduction the pipenetwork build reports while preserving voice quality
(pipenetwork quantized adaln to Q4 in a later revision — we keep bf16 precision by design).

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
    ):
        self.tables = tables
        self.timesteps = timesteps
        self.final_table = final_table
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

        Drop-in replacement for ``block.adaln_proj(t_emb)`` where ``t_emb`` was built from
        ``unique_t``.
        """
        idx = self._row_indices(unique_t)
        return tuple(mx.take(a, idx, axis=0) for a in self.tables[block_idx])

    def final_layer_gather(self, unique_t: Sequence[float]) -> Tuple[mx.array, mx.array]:
        """Return a 2-tuple ``(shift, scale)`` of ``[len(unique_t), hidden]`` arrays.

        Drop-in replacement for ``model.final_layer.adaln_proj(t_emb)``. If the cache was
        built without the final table (``build(..., cache_final=False)``) this raises.
        """
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
        dtype: mx.Dtype = mx.bfloat16,
        cache_final: bool = True,
        key_values: Optional[Sequence[float]] = None,
    ) -> "ModulationCache":
        """Precompute the modulation table for every DiT block from a running DiT.

        Args:
            dit: a ``MiniMaxH3Model`` whose ``adaln_proj`` weights are still loaded (bf16
                for a HEAVY_SUFFIX Q4 build).
            timesteps: ``(T,)`` float32 sorted-ascending union of distinct unique-t values,
                typically the output of :func:`schedule_timesteps`.
            dtype: storage dtype of the cache. bf16 halves the footprint and matches the
                precision the modulation is consumed at inside the block stack.
            cache_final: also cache ``final_layer.adaln_proj`` (default True).
        """
        # ``TimeEmbedder`` expects ``t: [M]`` float in [0, 1] and returns ``[M, time_embed_dim]``
        # (fp32). Cast to the AdalnProj linear's weight dtype implicitly via the linear call —
        # AdalnProj already handles that ``.astype(...)`` internally.
        t_emb = dit.time_embedder(timesteps)
        mx.eval(t_emb)

        tables: List[Tuple[mx.array, ...]] = []
        for block in dit.blocks:
            table = tuple(a.astype(dtype) for a in block.adaln_proj(t_emb))
            mx.eval(table)
            tables.append(table)

        final_table: Optional[Tuple[mx.array, mx.array]] = None
        if cache_final:
            final = tuple(a.astype(dtype) for a in dit.final_layer.adaln_proj(t_emb))
            mx.eval(final)
            final_table = final

        return cls(tables, timesteps, final_table=final_table, key_values=key_values)


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
