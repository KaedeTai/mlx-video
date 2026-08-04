"""H3 DiT building blocks: TimeEmbedder, AdalnProj, TokenRefiner, DiTBlock, FinalLayer.

Port of ``comfy/ldm/minimax/model.py`` lines 114-279.

Key MLX adaptations
-------------------
- **In-place → functional.** PyTorch uses ``.add_``/``.mul_``/``.addcmul_``
  on the residual stream; MLX arrays are immutable so every step returns a
  new array.
- **Segment-indexed modulation via slice+concat.** MLX has no item
  assignment; ``_mod_scale_shift`` / ``_mod_gate`` walk the (contiguous)
  segment list, slice the stream, modulate each piece, and concatenate.
- **SwiGLU MLP.** ``fc1`` outputs ``2*ffn``; we split into (gate, up),
  compute ``silu(gate) * up``, then ``fc2``.
- **fp32 output heads.** ``FinalLayer.{video,audio}_out`` are stored as
  fp32; we cast the pre-linear activation to fp32 before applying them.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn

from .attention import H3Attention, H3RMSNorm


# ---------------------------------------------------------------------------
# TimeEmbedder — sinusoidal + 2-layer MLP, fp32 throughout
# ---------------------------------------------------------------------------


class TimeEmbedder(nn.Module):
    """Sinusoidal timestep embedding + silu(Linear) + Linear.

    Reference stores biases in fp32 and outputs the same. We match that here.

    Weights:
      proj_in.{weight,bias}   [hidden, freq_dim] / [hidden]
      proj_out.{weight,bias}  [out_dim, hidden]  / [out_dim]
    """

    def __init__(self, freq_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.freq_dim = freq_dim
        self.proj_in = nn.Linear(freq_dim, hidden, bias=True)
        self.proj_out = nn.Linear(hidden, out_dim, bias=True)

    def __call__(self, t: mx.array) -> mx.array:
        """t: [M] in [0, 1] → [M, out_dim] (fp32)."""
        half = self.freq_dim // 2
        # exp(-log(10000) * arange(half) / half)
        freqs = mx.exp(-math.log(10000.0) * mx.arange(half, dtype=mx.float32) / half)
        args = t.astype(mx.float32)[:, None] * freqs[None, :]
        # NOTE: cos before sin (reference convention)
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        return self.proj_out(nn.silu(self.proj_in(emb)))


# ---------------------------------------------------------------------------
# AdalnProj — silu → Linear → chunk into `expand` per-modality tensors
# ---------------------------------------------------------------------------


class AdalnProj(nn.Module):
    """AdaLN projection: [M, t_dim] → tuple of ``expand`` tensors [M*modalities, hidden].

    Reference does ``x.view(M*modalities, expand*hidden).chunk(expand, dim=-1)``,
    so the last-axis chunking yields one scale/shift/gate per position.
    """

    def __init__(self, t_dim: int, hidden: int, expand: int, modalities: int,
                 apply_silu: bool = True):
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden = hidden
        self.apply_silu = apply_silu
        self.linear = nn.Linear(t_dim, expand * hidden * modalities, bias=True)

    def __call__(self, t_emb: mx.array) -> Tuple[mx.array, ...]:
        x = self.linear(nn.silu(t_emb) if self.apply_silu else t_emb)
        M = x.shape[0]
        x = x.reshape(M * self.modalities, self.expand * self.hidden)
        return tuple(mx.split(x, self.expand, axis=-1))


# ---------------------------------------------------------------------------
# Segment-indexed modulation helpers (functional analogues of the PyTorch in-place ops)
# ---------------------------------------------------------------------------


def _mod_scale_shift(h: mx.array, shift: mx.array, scale: mx.array,
                     segments: Sequence[Tuple[int, int, int]]) -> mx.array:
    """Per-segment ``h[a:b] * (1 + scale[row]) + shift[row]``.

    Segments are guaranteed to be contiguous and cover [0, seq_len).
    """
    parts = []
    for a, b, row in segments:
        s = shift[row:row + 1]
        sc = scale[row:row + 1]
        parts.append(h[a:b] * (1.0 + sc.astype(h.dtype)) + s.astype(h.dtype))
    return mx.concatenate(parts, axis=0)


def _mod_gate(x: mx.array, gate: mx.array, other: mx.array,
              segments: Sequence[Tuple[int, int, int]]) -> mx.array:
    """Per-segment ``x[a:b] + other[a:b] * gate[row]`` (gated residual add).

    Segments are contiguous and cover the full stream.
    """
    parts = []
    for a, b, row in segments:
        g = gate[row:row + 1].astype(x.dtype)
        parts.append(x[a:b] + other[a:b] * g)
    return mx.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# MLP — fused SwiGLU (fc1 outputs 2*ffn; split → silu(gate)*up → fc2)
# ---------------------------------------------------------------------------


class H3MLP(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.ffn = ffn
        self.fc1 = nn.Linear(hidden, ffn * 2, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.fc1(x)
        gate, up = mx.split(y, 2, axis=-1)
        return self.fc2(nn.silu(gate) * up)


# ---------------------------------------------------------------------------
# RefinerBlock & TokenRefiner (unmodulated — pre-DiT text refinement)
# ---------------------------------------------------------------------------


class RefinerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int, ffn: int,
                 eps: float, qk_eps: float):
        super().__init__()
        self.norm1 = H3RMSNorm(hidden, eps=eps)
        self.norm2 = H3RMSNorm(hidden, eps=eps)
        self.attn = H3Attention(hidden, heads, head_dim, qk_eps=qk_eps)
        self.mlp = H3MLP(hidden, ffn)

    def __call__(self, x: mx.array) -> mx.array:
        # No rope, no modulation — text refiner
        x = x + self.attn(self.norm1(x), rope_table=None)
        x = x + self.mlp(self.norm2(x))
        return x


class TokenRefiner(nn.Module):
    def __init__(self, num_layers: int, hidden: int, heads: int, head_dim: int, ffn: int,
                 eps: float, qk_eps: float, final_eps: float):
        super().__init__()
        self.blocks = [
            RefinerBlock(hidden, heads, head_dim, ffn, eps, qk_eps)
            for _ in range(num_layers)
        ]
        self.final_norm = H3RMSNorm(hidden, eps=final_eps)

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# DiTBlock — the main packed-stream block with adaLN modulation and rope
# ---------------------------------------------------------------------------


class DiTBlock(nn.Module):
    """The main H3 transformer block.

    Layout matches the reference: adaLN produces 6 tensors (shift/scale/gate for
    both attn and mlp), each covering ``M * 3`` rows (3 modality tags).
    """

    def __init__(self, hidden: int, heads: int, head_dim: int, ffn: int,
                 t_dim: int, eps: float, qk_eps: float, apply_silu: bool = True):
        super().__init__()
        self.norm1 = H3RMSNorm(hidden, eps=eps)
        self.norm2 = H3RMSNorm(hidden, eps=eps)
        self.attn = H3Attention(hidden, heads, head_dim, qk_eps=qk_eps)
        self.mlp = H3MLP(hidden, ffn)
        self.adaln_proj = AdalnProj(t_dim, hidden, expand=6, modalities=3, apply_silu=apply_silu)

    def __call__(
        self,
        x: mx.array,
        t_emb: mx.array,
        mod_segments: Sequence[Tuple[int, int, int]],
        rope_table: mx.array,
    ) -> mx.array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        h = _mod_scale_shift(self.norm1(x), shift_msa, scale_msa, mod_segments)
        x = _mod_gate(x, gate_msa, self.attn(h, rope_table=rope_table), mod_segments)
        h = _mod_scale_shift(self.norm2(x), shift_mlp, scale_mlp, mod_segments)
        return _mod_gate(x, gate_mlp, self.mlp(h), mod_segments)


# ---------------------------------------------------------------------------
# FinalLayer — adaln, then per-stream fp32 linear heads
# ---------------------------------------------------------------------------


class FinalLayer(nn.Module):
    """Final layer: modulate stream, cast to fp32, project to video/audio dims.

    The two output linears are stored in fp32 in the checkpoint (matches the
    reference: ``dtype=torch.float32``). We cast just the two target-stream
    slices to fp32 before applying them.
    """

    def __init__(self, hidden: int, t_dim: int, video_dim: int, audio_dim: int,
                 eps: float, apply_silu: bool = True):
        super().__init__()
        self.norm = H3RMSNorm(hidden, eps=eps)
        self.adaln_proj = AdalnProj(t_dim, hidden, expand=2, modalities=1, apply_silu=apply_silu)
        self.video_out = nn.Linear(hidden, video_dim, bias=True)
        self.audio_out = nn.Linear(hidden, audio_dim, bias=True)

    def __call__(
        self,
        x: mx.array,
        t_emb: mx.array,
        video_seg: Tuple[int, int, int],
        audio_seg: Tuple[int, int, int],
    ) -> Tuple[mx.array, mx.array]:
        shift, scale = self.adaln_proj(t_emb)
        va, vb, vrow = video_seg
        aa, ab, arow = audio_seg
        v_scale = 1.0 + scale[vrow:vrow + 1]
        a_scale = 1.0 + scale[arow:arow + 1]
        v_shift = shift[vrow:vrow + 1]
        a_shift = shift[arow:arow + 1]
        hv = self.norm(x[va:vb]) * v_scale.astype(x.dtype) + v_shift.astype(x.dtype)
        ha = self.norm(x[aa:ab]) * a_scale.astype(x.dtype) + a_shift.astype(x.dtype)
        hv = hv.astype(mx.float32)
        ha = ha.astype(mx.float32)
        return self.video_out(hv), self.audio_out(ha)
