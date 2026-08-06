"""Layer-group eviction manager for MiniMax H3 DiT (Apple Silicon / MLX).

Motivation
----------
On Apple Silicon (unified memory), all mx.array buffers count against
Metal's working-set / wired budget. A fully-loaded 50-block DiT with
Turbo LoRA overlay is ~40 GB wired weights, which leaves too little
head-room for activation buffers on longer sequences.

CUDA-style layer offloading is not directly available, but MLX does
release Metal buffers when the last mx.array reference is dropped and
``mx.clear_cache()`` is called. This module leverages that:

    - Split ``dit.blocks[:N]`` into contiguous groups (default 10).
    - Evict every group except the currently-executing one to CPU-side
      numpy arrays (regular pageable RAM, not Metal-wired).
    - Between block-groups, force ``mx.eval`` on the running activation
      so the just-executed weights can be safely evicted.

The evicted state is a Python dict of numpy arrays keyed by the same
parameter paths that ``mlx.utils.tree_flatten`` produces, so both the
base Q4 packed weights (``uint32``) and the Turbo LoRA A/B tensors
(``bfloat16``, stored via a uint16 bit-view) round-trip losslessly.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten


# ---------------------------------------------------------------------------
# bf16 <-> numpy uint16 bit-view helpers
# ---------------------------------------------------------------------------


_BF16_TAG = "bfloat16"


def _mx_to_np(v: mx.array):
    """Copy an mx.array to a numpy array. Uses uint16 bit-view for bf16."""
    mx.eval(v)
    if v.dtype == mx.bfloat16:
        return np.array(v.view(mx.uint16), copy=True), _BF16_TAG
    return np.array(v, copy=True), None


def _np_to_mx(arr: np.ndarray, tag) -> mx.array:
    if tag == _BF16_TAG:
        return mx.array(arr).view(mx.bfloat16)
    return mx.array(arr)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class LayerGroupManager:
    """Group-wise offload of ``dit.blocks`` weights.

    Parameters
    ----------
    dit_model : MiniMaxH3Model
        Fully loaded DiT (weights + optional Turbo LoRA overlays already
        installed).  The constructor immediately evicts all groups except
        those listed in ``keep_hot`` so the up-front wired footprint drops
        to roughly ``group_size / num_layers`` of the weights.
    group_size : int
        Number of contiguous blocks per group.
    keep_hot : Sequence[int]
        Group indices to leave in Metal buffers at construction time
        (default: empty — evict every group).  Useful for warm-start.
    verbose : bool
        Print eviction / activation timings.
    """

    def __init__(
        self,
        dit_model,
        group_size: int = 10,
        keep_hot: Sequence[int] = (),
        verbose: bool = False,
    ):
        self.dit = dit_model
        n = len(dit_model.blocks)
        self.total_blocks = n
        self.group_size = int(group_size)
        assert self.group_size > 0
        self.num_groups = (n + self.group_size - 1) // self.group_size
        self.group_ranges: List[tuple] = [
            (i * self.group_size, min((i + 1) * self.group_size, n))
            for i in range(self.num_groups)
        ]
        # dormant[gi] = list per-block of dicts {param_path: (np.ndarray, dtype_tag)}
        self.dormant: List = [None] * self.num_groups
        self.hot: set = set(range(self.num_groups))
        self.verbose = bool(verbose)
        self._n_activations = 0
        self._n_evictions = 0
        self._bytes_dormant = 0
        for gi in range(self.num_groups):
            if gi not in keep_hot:
                self.evict_group(gi)
        if self.verbose:
            gb = self._bytes_dormant / 1024**3
            print(f"[LGE] init: {self.num_groups} groups of size ~{self.group_size}, "
                  f"dormant weight footprint {gb:.2f} GB "
                  f"(active_mlx={mx.get_active_memory()/1024**3:.2f} GB)")

    # ------------------------------------------------------------------
    # eviction / activation
    # ------------------------------------------------------------------

    def evict_group(self, gi: int) -> None:
        if gi not in self.hot:
            return
        s, e = self.group_ranges[gi]
        per_block = []
        for bi in range(s, e):
            blk = self.dit.blocks[bi]
            flat = tree_flatten(blk.parameters())
            dormant_block = {}
            placeholder = []
            for k, v in flat:
                np_arr, tag = _mx_to_np(v)
                dormant_block[k] = (np_arr, tag)
                self._bytes_dormant += np_arr.nbytes
                placeholder.append((k, mx.zeros((0,), dtype=v.dtype)))
            blk.update(tree_unflatten(placeholder))
            per_block.append(dormant_block)
        self.dormant[gi] = per_block
        self.hot.discard(gi)
        self._n_evictions += 1
        mx.clear_cache()

    def activate_group(self, gi: int) -> None:
        if gi in self.hot:
            return
        s, e = self.group_ranges[gi]
        per_block = self.dormant[gi]
        assert per_block is not None, f"group {gi} has no dormant state"
        for local_bi, bi in enumerate(range(s, e)):
            blk = self.dit.blocks[bi]
            dormant_block = per_block[local_bi]
            restored = [(k, _np_to_mx(np_arr, tag))
                        for k, (np_arr, tag) in dormant_block.items()]
            blk.update(tree_unflatten(restored))
            # Force materialization now so first block call doesn't stall
            # on a huge weight-upload spike.
            mx.eval(*[v for _, v in restored])
            self._bytes_dormant -= sum(a.nbytes for a, _ in dormant_block.values())
        self.dormant[gi] = None
        self.hot.add(gi)
        self._n_activations += 1

    @contextmanager
    def active_group(self, gi: int) -> Iterator[List[nn.Module]]:
        """Activate group ``gi``, yield the block list, then evict."""
        self.activate_group(gi)
        s, e = self.group_ranges[gi]
        try:
            yield [self.dit.blocks[bi] for bi in range(s, e)]
        finally:
            self.evict_group(gi)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "num_groups": self.num_groups,
            "group_size": self.group_size,
            "total_blocks": self.total_blocks,
            "hot_groups": sorted(self.hot),
            "dormant_bytes": self._bytes_dormant,
            "dormant_gb": self._bytes_dormant / 1024**3,
            "n_activations": self._n_activations,
            "n_evictions": self._n_evictions,
            "active_mlx_gb": mx.get_active_memory() / 1024**3,
        }
