"""H3 attention: RMSNorm-per-head + split-half rotary + MHA.

Port of ``Attention`` in ``comfy/ldm/minimax/model.py`` (lines 141-174).

The reference uses ``comfy.quant_ops.ck.rms_rope_split_half`` (a fused
Metal/CUDA kernel that RMSNorm's q/k per head and applies rotary in place).
We implement the plain functional equivalent:

  1. QKV projection (fused ``qkv_proj`` linear)
  2. reshape into (S, H, head_dim)
  3. RMSNorm on q/k over head_dim
  4. split-half rotary on the first ``rot_dim`` slice of head_dim
  5. mx.fast.scaled_dot_product_attention (single batch axis)
  6. out_proj

The packed sequence has no explicit batch dim (batch is always 1 in H3), so we
carry a leading axis of size 1 through the attention step to keep the API in
line with mx.fast.sdpa's expectations of ``[B, H, L, D]``.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .rope import apply_split_half_rope


class H3RMSNorm(nn.Module):
    """RMSNorm with a learnable scale over the last axis."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class H3Attention(nn.Module):
    """Fused-qkv self-attention with per-head qk-RMSNorm and split-half rope.

    Weights:
      qkv_proj.weight  [3*inner, hidden]         (bias=False)
      q_norm.weight    [head_dim]
      k_norm.weight    [head_dim]
      out_proj.weight  [hidden, inner]            (bias=False)
    """

    def __init__(self, hidden: int, heads: int, head_dim: int, qk_eps: float = 1e-5):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = head_dim
        self.inner = heads * head_dim

        self.qkv_proj = nn.Linear(hidden, self.inner * 3, bias=False)
        self.q_norm = H3RMSNorm(head_dim, eps=qk_eps)
        self.k_norm = H3RMSNorm(head_dim, eps=qk_eps)
        self.out_proj = nn.Linear(self.inner, hidden, bias=False)

        self.scale = 1.0 / (head_dim ** 0.5)

    def __call__(
        self,
        x: mx.array,
        rope_table: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Self-attention.

        Args
        ----
        x          : [S, hidden]
        rope_table : [1, S, 1, half, 2, 2] rotation table (see rope.py), or None
        mask       : optional attention mask [S, S] or broadcastable
        """
        s = x.shape[0]
        H, D = self.heads, self.head_dim
        qkv = self.qkv_proj(x)                           # [S, 3*inner]
        q, k, v = mx.split(qkv, 3, axis=-1)              # each [S, inner]
        q = q.reshape(s, H, D)
        k = k.reshape(s, H, D)
        v = v.reshape(s, H, D)

        # Per-head RMSNorm on q/k
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Split-half rotary on the first rot_dim slice
        if rope_table is not None:
            rot_dim = rope_table.shape[-3] * 2
            q = apply_split_half_rope(q, rope_table, rot_dim)
            k = apply_split_half_rope(k, rope_table, rot_dim)

        # mx.fast.scaled_dot_product_attention expects [B, H, S, D]
        q = q.transpose(1, 0, 2)[None, ...]   # [1, H, S, D]
        k = k.transpose(1, 0, 2)[None, ...]
        v = v.transpose(1, 0, 2)[None, ...]

        if mask is not None:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        else:
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # [1, H, S, D] -> [S, inner]
        out = out[0].transpose(1, 0, 2).reshape(s, self.inner)
        return self.out_proj(out)
