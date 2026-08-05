"""MiniMax H3 Video VAE — MLX port.

3D causal CNN encoder (EncoderFCN3D) + 36-layer ViT3D decoder.

Reference: comfy/ldm/minimax/vae.py (694 LOC) and
`~/models/MiniMax-H3-raw/Ref2VA/video_vae/*.py`.

Tensor layout convention
------------------------
Internally we use MLX's channel-last layout throughout:

    5-D "video" tensors:  (B, T, H, W, C)     (== PyTorch NCDHW.permute(0,2,3,4,1))
    3-D "token" tensors:  (B, N, C)

The public :meth:`MiniMaxH3VideoVAE.encode`/:meth:`decode` accept and return
NCDHW arrays to match the reference and the converter's expectation, doing
the permute at the boundary.

Weight naming is 1:1 with the source safetensors so a bare state-dict load
just works after Conv3d weight-layout transposition (handled by
``convert.py``: PyTorch ``(O, I, D, H, W)`` -> MLX ``(O, D, H, W, I)``).

Phase 2 scope: single-clip forward path (no spatial tiling, no multi-chunk
temporal). ``clip_length=17`` frames @ up to 384x384 fits comfortably in
unified memory and gives us a clean parity check against the reference.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Latent / pixel normalization tables (copied from the checkpoint config)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]

LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.68317747116088865, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.96531379222869875, 1.05698859691619875,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049925, 0.7197399735450745, 0.69362932443618775,
    2.961095094680786, 2.7694199085235595, 3.0496184825897215, 2.1088054180145265,
    3.276226282119751, 3.1627357006073, 2.28168129920959475, 2.6127843856811525,
]


# ---------------------------------------------------------------------------
# Padding helpers (MLX only supplies constant/edge; we need reflect + causal)
# ---------------------------------------------------------------------------


def _reflect_pad_hw(x: mx.array, pad_h: int, pad_w: int) -> mx.array:
    """Symmetric reflect padding on the H and W axes of an NDHWC tensor.

    PyTorch semantics: the boundary element is NOT duplicated, so padding a
    length-N row with pad=1 uses index 1 on the left and index N-2 on the right.
    MLX doesn't have ``mx.flip``, but negative-step slicing works and returns
    a reversed view.
    """
    if pad_h > 0:
        top = x[:, :, 1:pad_h + 1, :, :][:, :, ::-1, :, :]
        bot = x[:, :, -pad_h - 1:-1, :, :][:, :, ::-1, :, :]
        x = mx.concatenate([top, x, bot], axis=2)
    if pad_w > 0:
        left = x[:, :, :, 1:pad_w + 1, :][:, :, :, ::-1, :]
        right = x[:, :, :, -pad_w - 1:-1, :][:, :, :, ::-1, :]
        x = mx.concatenate([left, x, right], axis=3)
    return x


def _reflect_pad_hw_bottom_right(x: mx.array, pad_h_br: int, pad_w_br: int) -> mx.array:
    """Asymmetric reflect padding: pad only the *bottom* of H and the *right* of W."""
    if pad_h_br > 0:
        bot = x[:, :, -pad_h_br - 1:-1, :, :][:, :, ::-1, :, :]
        x = mx.concatenate([x, bot], axis=2)
    if pad_w_br > 0:
        right = x[:, :, :, -pad_w_br - 1:-1, :][:, :, :, ::-1, :]
        x = mx.concatenate([x, right], axis=3)
    return x


def _causal_pad_t(x: mx.array, pad_t: int) -> mx.array:
    """Front-only zero padding along the T (axis=1) dimension of NDHWC."""
    if pad_t <= 0:
        return x
    B, _, H, W, C = x.shape
    zeros = mx.zeros((B, pad_t, H, W, C), dtype=x.dtype)
    return mx.concatenate([zeros, x], axis=1)


# ---------------------------------------------------------------------------
# CNN encoder building blocks
# ---------------------------------------------------------------------------


class CausalConv3d(nn.Module):
    """3D convolution with reflect spatial padding and causal (zero) time padding.

    The stored weight has MLX layout ``(O, D, H, W, I)`` so that a converted
    PyTorch weight (``permute(0, 2, 3, 4, 1)``) drops straight in.

    Input/output layout: NDHWC.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        # ``causal_padding`` mirrors the reference field name; index 0 is time.
        self.causal_padding = tuple(padding)

        # Weight shape matches MLX's Conv3d layout so mx.conv_general is happy.
        self.weight = mx.zeros(
            (
                out_channels,
                self.kernel_size[0],
                self.kernel_size[1],
                self.kernel_size[2],
                in_channels,
            )
        )
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        pad_t, pad_h, pad_w = self.causal_padding

        if pad_t == 0 and pad_h == 0 and pad_w == 0:
            # nin_shortcut / 1x1x1 conv path: no padding, straight conv.
            return mx.conv_general(
                x,
                self.weight,
                stride=self.stride,
                padding=0,
            ) + self.bias

        # Spatial reflect padding first.
        if pad_h > 0 or pad_w > 0:
            x = _reflect_pad_hw(x, pad_h, pad_w)

        # Then causal temporal front-pad. For D==1 we still prepend pad_t*2
        # zeros; the reference short-circuits into an "autopad='causal_zero'"
        # path that is functionally identical (both leave a k-frame receptive
        # field where the first k-1 taps are zero).
        x = _causal_pad_t(x, pad_t * 2)

        return mx.conv_general(
            x,
            self.weight,
            stride=self.stride,
            padding=0,
        ) + self.bias


class TemporalIsolatedGroupNorm(nn.Module):
    """GroupNorm over C at each frame independently.

    Matches the reference's ``TemporalIsolatedGroupNorm`` when 5-D input is
    supplied: reshape (B, T, H, W, C) -> (B*T, H, W, C), GroupNorm, then
    unfold. Statistics are per-frame, per-group.
    """

    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.weight = mx.ones((num_channels,))
        self.bias = mx.zeros((num_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        x_flat = x.reshape(B * T, H, W, C)
        # Manual GroupNorm across (H, W, C_in_group) with fp32 accumulate.
        y = x_flat.astype(mx.float32)
        y = y.reshape(B * T, H * W, self.num_groups, C // self.num_groups)
        mean = y.mean(axis=(1, 3), keepdims=True)
        var = ((y - mean) ** 2).mean(axis=(1, 3), keepdims=True)
        y = (y - mean) * mx.rsqrt(var + self.eps)
        y = y.reshape(B * T, H, W, C)
        y = y * self.weight + self.bias
        y = y.astype(x.dtype)
        return y.reshape(B, T, H, W, C)


class Downsample3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_stride: int = 1,
        space_stride: int = 2,
    ):
        super().__init__()
        self.space_stride = space_stride
        self.time_stride = time_stride
        # Padding tuple is (t, h, w); we let CausalConv3d handle time via causal
        # front-pad. Spatial pad here is 0 because Downsample3D pre-pads
        # asymmetrically (0,1 on H,W) below.
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=(1, 0, 0),
            stride=(time_stride, space_stride, space_stride),
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.space_stride == 2:
            x = _reflect_pad_hw_bottom_right(x, pad_h_br=1, pad_w_br=1)
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = TemporalIsolatedGroupNorm(in_channels)
        self.norm2 = TemporalIsolatedGroupNorm(out_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = CausalConv3d(
                in_channels, out_channels, kernel_size=1
            )
        else:
            self.nin_shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return h + x


class _DownLevel(nn.Module):
    """Container that carries ``block`` (ModuleList) and optional ``downsample``.

    Exists so that the checkpoint keys ``encoder.down.{i}.block.{j}.*`` map
    directly onto attributes on an MLX Module.
    """

    def __init__(self):
        super().__init__()
        self.block: list = []
        self.downsample: Optional[Downsample3D] = None

    def __call__(self, x: mx.array) -> mx.array:
        for blk in self.block:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class EncoderFCN3D(nn.Module):
    def __init__(
        self,
        ch: int,
        ch_mult: Sequence[int],
        space_down: Sequence[int],
        time_down: Sequence[int],
        num_res_blocks: int,
        in_channels: int,
        z_channels: int,
        double_z: bool = True,
    ):
        super().__init__()
        self.num_levels = len(ch_mult)
        num_res_blocks_list = [num_res_blocks] * self.num_levels

        block_mid = [ch * ch_mult[i] for i in range(self.num_levels)]
        block_in = [block_mid[0]] + block_mid[:-1]

        self.conv_in = CausalConv3d(in_channels, block_in[0], kernel_size=3, padding=1)

        self.down: list = []
        for i_level in range(self.num_levels):
            level = _DownLevel()
            for i in range(num_res_blocks_list[i_level]):
                level.block.append(
                    ResnetBlock3D(
                        in_channels=block_in[i_level] if i == 0 else block_mid[i_level],
                        out_channels=block_mid[i_level],
                    )
                )
            if space_down[i_level] * time_down[i_level] > 1:
                level.downsample = Downsample3D(
                    block_mid[i_level],
                    block_mid[i_level],
                    time_stride=time_down[i_level],
                    space_stride=space_down[i_level],
                )
            self.down.append(level)

        self.norm_out = TemporalIsolatedGroupNorm(block_mid[-1])
        self.conv_out = CausalConv3d(
            block_mid[-1],
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv_in(x)
        for level in self.down:
            h = level(h)
        h = nn.silu(self.norm_out(h))
        return self.conv_out(h)


class PointwiseConv3d(nn.Module):
    """1x1x1 3D convolution (used by quant_conv / post_quant_conv).

    A pointwise conv is equivalent to a Linear over the channel dim, but we
    keep the Conv3d module + weight name so the checkpoint keys line up.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = mx.zeros((out_channels, 1, 1, 1, in_channels))
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.conv_general(x, self.weight, stride=1, padding=0) + self.bias


# ---------------------------------------------------------------------------
# ViT3D decoder building blocks
# ---------------------------------------------------------------------------


def _rms_norm(x: mx.array, weight: Optional[mx.array], eps: float) -> mx.array:
    """RMSNorm along the last dim. Weight may be None (no affine)."""
    dtype = x.dtype
    x32 = x.astype(mx.float32)
    x32 = x32 * mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    if weight is not None:
        x32 = x32 * weight.astype(mx.float32)
    return x32.astype(dtype)


class RMSNormAffine(nn.Module):
    """RMSNorm with a learnable per-channel scale, matches ``nn.RMSNorm(..., affine=True)``."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return _rms_norm(x, self.weight, self.eps)


class LayerNormAffine(nn.Module):
    """LayerNorm with learnable weight + bias along the last dim."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))
        self.bias = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x32 = x.astype(mx.float32)
        mean = x32.mean(axis=-1, keepdims=True)
        var = ((x32 - mean) ** 2).mean(axis=-1, keepdims=True)
        x32 = (x32 - mean) * mx.rsqrt(var + self.eps)
        x32 = x32 * self.weight.astype(mx.float32) + self.bias.astype(mx.float32)
        return x32.astype(dtype)


class RotaryEmbeddingND(nn.Module):
    """Rotary positional embedding over N spatial-time axes.

    Produces ``(cos, sin)`` tables of shape ``(B, N_tokens, 1, rope_dim)``.
    Matches the Ref2VA implementation (``use_angle=True`` -> angle_scale=2π).
    """

    def __init__(self, dim: int, rotary_base: float = 100.0, n_dim: int = 3):
        super().__init__()
        if dim % (2 * n_dim) != 0:
            raise ValueError(f"dim {dim} must be divisible by 2 * n_dim ({2 * n_dim})")
        self.dim = dim
        self.n_dim = n_dim
        self.angle_scale = 2.0 * math.pi

        # inv_freq: 1 / base ** arange(0, 1, 2*n/dim). Length = dim / (2 * n_dim).
        exponents = mx.arange(0.0, 1.0, 2.0 * n_dim / dim, dtype=mx.float32)
        self.inv_freq = 1.0 / (rotary_base ** exponents)

    def __call__(self, img_ids: mx.array):
        # img_ids: (B, N, n_dim)  in [-1, 1]
        angles = (
            self.angle_scale
            * img_ids[:, :, :, None]
            * self.inv_freq[None, None, None, :]
        )
        # (B, N, n_dim, freqs) -> (B, N, n_dim*freqs)
        B, N, nd, f = angles.shape
        angles = angles.reshape(B, N, nd * f)
        # tile(2) doubles the last dim (rotate_half operates on two halves)
        angles = mx.concatenate([angles, angles], axis=-1)
        # insert head dim
        angles = angles[:, :, None, :]
        return mx.cos(angles), mx.sin(angles)


def _rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rope(t: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply RoPE to the leading ``rot_dim`` channels of ``t``.

    ``t``   shape (B, N, H, D)
    ``cos`` shape (B, N, 1, rot_dim)  broadcasts across the head axis.
    """
    rot_dim = cos.shape[-1]
    t_dim = t.shape[-1]
    cos = cos.astype(t.dtype)
    sin = sin.astype(t.dtype)
    if rot_dim < t_dim:
        t_rot = t[..., :rot_dim]
        t_pass = t[..., rot_dim:]
        t_rot = t_rot * cos + _rotate_half(t_rot) * sin
        return mx.concatenate([t_rot, t_pass], axis=-1)
    return t * cos + _rotate_half(t) * sin


class GatedFeedForward(nn.Module):
    """SwiGLU FFN: ``w2(silu(gate) * up)`` with fused ``w1: dim -> 2 * inner``."""

    def __init__(self, dim: int, mult: int = 4, bias: bool = True):
        super().__init__()
        inner = dim * mult
        self.w1 = nn.Linear(dim, inner * 2, bias=bias)
        self.w2 = nn.Linear(inner, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.w1(x)
        gate, up = mx.split(h, 2, axis=-1)
        return self.w2(nn.silu(gate) * up)


class ViTAttention(nn.Module):
    """Multi-head self-attention with fused QKV projection, RoPE, and RMS QK norm."""

    def __init__(self, heads: int, dim_head: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.eps = eps
        dim = heads * dim_head
        # qk_norm_affine=False in Ref2VA: RMSNorm without learnable weight.
        # We keep the eps but do not allocate a weight tensor -- no checkpoint keys.
        self.to_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.to_out = nn.Linear(dim, dim, bias=bias)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        B, N, _ = x.shape
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(B, N, self.heads, 3 * self.dim_head)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = _rms_norm(q, None, self.eps)
        k = _rms_norm(k, None, self.eps)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # (B, N, H, D) -> (B, H, N, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.dim_head ** -0.5
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, N, self.heads * self.dim_head)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, heads: int, dim_head: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        dim = heads * dim_head
        self.norm1 = RMSNormAffine(dim, eps=eps)
        self.attn = ViTAttention(heads=heads, dim_head=dim_head, bias=bias, eps=eps)
        self.scale1 = mx.zeros((dim,))
        self.norm2 = RMSNormAffine(dim, eps=eps)
        self.ff = GatedFeedForward(dim=dim, bias=bias)
        self.scale2 = mx.zeros((dim,))

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        # Reference upcasts the residual stream to fp32 before every norm.
        x = x + self.scale1 * self.attn(self.norm1(x), cos, sin)
        x = x + self.scale2 * self.ff(self.norm2(x))
        return x


def _create_token_ids(patch_dims, dtype=mx.float32) -> mx.array:
    """3-D length-normalized token ids in [-1, 1], flattened to (1, N, 3)."""
    coords_list = []
    for d in patch_dims:
        c = (mx.arange(0.5, d, dtype=dtype) / d) * 2.0 - 1.0
        coords_list.append(c)
    # meshgrid indexing='ij'
    grids = mx.meshgrid(*coords_list, indexing="ij")
    coords = mx.stack(grids, axis=-1)  # (D0, D1, D2, 3)
    return coords.reshape(1, -1, len(patch_dims))


class ViT3DDecoder(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        patch_size_t: int = 4,
        in_channels: int = 24,
        out_channels: int = 3,
        num_layers: int = 36,
        heads: int = 32,
        dim_head: int = 64,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.75,
        bias: bool = True,
        eps: float = 1e-5,
        num_register_tokens: int = 4,
    ):
        super().__init__()
        dim = heads * dim_head
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.pos_embed = RotaryEmbeddingND(
            int(dim_head * rope_dim_ratio), rope_theta, n_dim=3
        )
        self.x_embedder = nn.Linear(in_channels, dim)
        self.register_tokens = mx.zeros((1, num_register_tokens, dim))
        # ``mask_token`` is stored as a checkpoint buffer even though it is
        # unused at inference. We hold it so state-dict load doesn't complain.
        self.mask_token = mx.zeros((1, 1, dim))

        self.transformer_blocks = [
            TransformerBlock(heads=heads, dim_head=dim_head, bias=bias, eps=eps)
            for _ in range(num_layers)
        ]

        self.norm_out = LayerNormAffine(dim, eps=eps)
        self.proj_out = nn.Linear(
            dim, out_channels * patch_size_t * patch_size * patch_size
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T_lat, H_lat, W_lat, C_in)  (NDHWC)
        B, latent_T, latent_H, latent_W, C = x.shape

        # Patchify (patch_size = patch_size_t = 1 for the ViT input; the latent
        # is already at patch resolution, we just flatten the 3 spatial-time
        # axes into a single token sequence).
        h = x.reshape(B, latent_T * latent_H * latent_W, C)
        h = self.x_embedder(h)

        num_patches = h.shape[1]
        num_suffix = 1 + self.num_register_tokens

        register = mx.broadcast_to(
            self.register_tokens, (B, self.num_register_tokens, h.shape[-1])
        )
        cls_placeholder = mx.zeros(
            (B, 1, h.shape[-1]), dtype=h.dtype
        )
        h = mx.concatenate([h, register.astype(h.dtype), cls_placeholder], axis=1)

        img_ids = _create_token_ids((latent_T, latent_H, latent_W), dtype=x.dtype)
        img_ids = mx.broadcast_to(img_ids, (B, img_ids.shape[1], 3))
        suffix_ids = mx.zeros((B, num_suffix, 3), dtype=img_ids.dtype)
        img_ids = mx.concatenate([img_ids, suffix_ids], axis=1)

        cos, sin = self.pos_embed(img_ids)

        for block in self.transformer_blocks:
            h = block(h, cos, sin)

        h = self.norm_out(h)
        out = self.proj_out(h)
        out = out[:, :num_patches, :]

        # Unpatchify: (B, T_lat*H_lat*W_lat, out_ch * pt * ph * pw)
        pt, ph, pw = self.patch_size_t, self.patch_size, self.patch_size
        oc = self.out_channels
        out = out.reshape(B, latent_T, latent_H, latent_W, oc, pt, ph, pw)
        # (B, T_lat, H_lat, W_lat, C, pt, ph, pw)
        # -> (B, T_lat, pt, H_lat, ph, W_lat, pw, C)   [NDHWC output layout]
        out = out.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        out = out.reshape(
            B,
            latent_T * pt,
            latent_H * ph,
            latent_W * pw,
            oc,
        )
        return out


# ---------------------------------------------------------------------------
# Full VAE
# ---------------------------------------------------------------------------


class MiniMaxH3VideoVAE(nn.Module):
    """Encode/decode a video clip with the MiniMax H3 VAE.

    Public API (``encode``/``decode``) operates on **NCDHW** tensors to match
    the reference implementation; internally everything is NDHWC.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_ch: int = 3,
        ch: int = 128,
        embed_dim: int = 24,
        z_channels: int = 24,
        ch_mult: Sequence[int] = (1, 2, 2, 4, 4, 8),
        num_res_blocks: int = 2,
        space_down: Sequence[int] = (2, 2, 2, 2, 1, 1),
        time_down: Sequence[int] = (1, 2, 2, 1, 1, 1),
        clip_length: int = 17,
        token_drop: int = 3,
        tile_size: int = 256,
        tile_overlap_min: int = 64,
        # Phase 8.11-1: ComfyUI's default is tiling=True (see comfy/ldm/minimax/vae.py
        # __init__). The spatial tile path linearly cross-fades ``tile_overlap_min``
        # pixels between adjacent tiles, so the per-16-px per-patch grid is
        # blended across a wider band once we cross a tile boundary. For
        # frames <= tile_size we skip tiling and take the single-shot path.
        tiling: bool = True,
        # Phase 8.8: deblock defaulted OFF. It was a post-decode 16-px
        # boundary low-pass (alpha=0.35, bw=3) added in Phase 8.7 to mask a
        # crosshatch texture from the ViT3D decoder. In practice it touches
        # ~60% of output pixels with a 35%-weight mirror blend — precisely
        # the "frosted glass" the user reported. The ComfyUI reference has
        # no such pass. The underlying grid must be fixed in the decoder
        # itself (or via proper tile_overlap in decode_temporal), not
        # papered over. Set deblock_patches=True to opt back into the
        # Phase 8.7 band-aid.
        deblock_patches: bool = False,
        deblock_blend_width: int = 3,
        deblock_alpha: float = 0.35,
    ):
        super().__init__()
        self.vae_ratio = int(math.prod(space_down))
        self.vae_ratio_t = int(math.prod(time_down))

        self.clip_length = clip_length
        self.token_drop = token_drop
        # Derived quantities for decode_temporal (Phase 8.9-a port).
        # Mirrors ComfyUI comfy/ldm/minimax/vae.py:346-351.
        self.tokens_chunk_size = int(math.ceil(clip_length / self.vae_ratio_t))
        self.frame_pre_padding = (-clip_length) % self.vae_ratio_t
        self.token_overlap = (-token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(
            self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0
        )
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap_min
        self.tiling = tiling
        # Phase 8.7: Post-decode deblock across VAE patch boundaries (16 px).
        # The ViT3DDecoder emits each 16x16 patch via a single Linear proj_out;
        # adjacent patches don't perfectly blend, leaving a crosshatch texture
        # on smooth regions (skin, backgrounds). PyTorch reference has the
        # same artifact (col=1.97 / row=2.28 boundary-edge ratio on a real
        # image round-trip). Enabling deblock deviates from bit-for-bit ref
        # match but is off-boundary sharpness-neutral (Laplacian var identical
        # after masking ±4 px around each boundary). Set deblock_patches=False
        # to match the reference exactly.
        self.deblock_patches = deblock_patches
        self.deblock_blend_width = deblock_blend_width
        self.deblock_alpha = deblock_alpha
        self.embed_dim = embed_dim
        self.z_channels = z_channels

        self.encoder = EncoderFCN3D(
            ch=ch,
            ch_mult=list(ch_mult),
            space_down=list(space_down),
            time_down=list(time_down),
            num_res_blocks=num_res_blocks,
            in_channels=in_channels,
            z_channels=z_channels,
            double_z=True,
        )
        self.quant_conv = PointwiseConv3d(z_channels * 2, 2 * embed_dim)
        self.post_quant_conv = PointwiseConv3d(embed_dim, z_channels)
        self.decoder = ViT3DDecoder(
            patch_size=self.vae_ratio,
            patch_size_t=self.vae_ratio_t,
            in_channels=z_channels,
            out_channels=out_ch,
        )

        # Buffers stored to match reference state_dict.
        self.latents_mean = mx.array(LATENTS_MEAN, dtype=mx.float32)
        self.latents_std = mx.array(LATENTS_STD, dtype=mx.float32)
        self.pixel_mean = mx.array(IMAGENET_MEAN, dtype=mx.float32).reshape(1, 1, 1, 1, 3)
        self.pixel_std = mx.array(IMAGENET_STD, dtype=mx.float32).reshape(1, 1, 1, 1, 3)

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _nchw_to_ndhwc(x: mx.array) -> mx.array:
        # (B, C, T, H, W) -> (B, T, H, W, C)
        return x.transpose(0, 2, 3, 4, 1)

    @staticmethod
    def _ndhwc_to_nchw(x: mx.array) -> mx.array:
        # (B, T, H, W, C) -> (B, C, T, H, W)
        return x.transpose(0, 4, 1, 2, 3)

    # ------------------------------------------------------------------
    # Core forward paths (no tiling, no multi-clip chunking -- Phase 2 scope)
    # ------------------------------------------------------------------

    def _encode_moments_ndhwc(self, x_ndhwc: mx.array) -> mx.array:
        return self.quant_conv(self.encoder(x_ndhwc))

    def _decode_pixels_ndhwc(self, z_ndhwc: mx.array) -> mx.array:
        return self.decoder(self.post_quant_conv(z_ndhwc))

    def _adaptive_encode_ndhwc(self, x_ndhwc: mx.array) -> mx.array:
        if self.tiling and (
            x_ndhwc.shape[2] > self.tile_size or x_ndhwc.shape[3] > self.tile_size
        ):
            return self.tiled_encode_ndhwc(x_ndhwc)
        return self._encode_moments_ndhwc(x_ndhwc)

    def _adaptive_decode_ndhwc(self, z_ndhwc: mx.array) -> mx.array:
        if self.tiling and (
            z_ndhwc.shape[2] * self.vae_ratio > self.tile_size
            or z_ndhwc.shape[3] * self.vae_ratio > self.tile_size
        ):
            return self.tiled_decode_ndhwc(z_ndhwc)
        return self._decode_pixels_ndhwc(z_ndhwc)

    # ------------------------------------------------------------------
    # Phase 8.11-1: spatial tiling.
    # Ported from ComfyUI comfy/ldm/minimax/vae.py:400-518. Splits the
    # (B, T, H, W, C) tensor into overlapping tiles along the H and W
    # axes (axis=2 and axis=3 in NDHWC), encodes/decodes each tile
    # separately, then linearly cross-fades the overlap regions.
    #
    # Overlap size is a multiple of ``vae_ratio`` so latent-space and
    # pixel-space grids stay aligned. Tile size is the pixel-space
    # length (encoder input / decoder output resolution).
    # ------------------------------------------------------------------
    def _split_tiles(self, input_len: int):
        tile_size = self.tile_size
        if tile_size >= input_len:
            return [0], [input_len], []

        N = int(math.ceil(input_len / tile_size))
        while True:
            overlaps = [self.tile_overlap_min] * (N - 1)
            remaining = tile_size * N - sum(overlaps) - input_len
            if remaining < 0:
                N += 1
            else:
                break

        # distribute leftover overlap in multiples of vae_ratio
        remaining_units = remaining // self.vae_ratio
        for i in range(remaining_units):
            overlaps[i % (N - 1)] += self.vae_ratio

        tile_start_idx = [0]
        for i in range(N - 1):
            tile_start_idx.append(tile_start_idx[-1] + tile_size - overlaps[i])
        return tile_start_idx, [tile_size] * N, overlaps

    @staticmethod
    def _blend_axis(a: "mx.array", b: "mx.array", blend_extent: int, axis: int) -> "mx.array":
        """Linear cross-fade of ``a``'s tail into ``b``'s head along ``axis``."""
        blend_extent = int(min(a.shape[axis], b.shape[axis], blend_extent))
        if blend_extent <= 0:
            return b
        pos = mx.arange(blend_extent, dtype=b.dtype)
        shape = [1] * b.ndim
        shape[axis] = blend_extent
        w_b = (pos / blend_extent).reshape(tuple(shape))
        w_a = 1.0 - w_b

        # slice a's tail and b's head
        a_slices = [slice(None)] * a.ndim
        a_slices[axis] = slice(-blend_extent, None)
        b_slices = [slice(None)] * b.ndim
        b_slices[axis] = slice(0, blend_extent)
        a_tail = a[tuple(a_slices)]
        b_head = b[tuple(b_slices)]
        blended = a_tail * w_a + b_head * w_b

        if blend_extent < b.shape[axis]:
            b_rest_slices = [slice(None)] * b.ndim
            b_rest_slices[axis] = slice(blend_extent, None)
            return mx.concatenate([blended, b[tuple(b_rest_slices)]], axis=axis)
        return blended

    def tiled_encode_ndhwc(self, x_ndhwc: mx.array) -> mx.array:
        """Overlapping spatial-tiled encode.

        ``x_ndhwc`` shape (B, T, H, W, C_in). Output shape
        (B, T_lat, H_lat, W_lat, 2*z_channels).
        """
        H, W = int(x_ndhwc.shape[2]), int(x_ndhwc.shape[3])
        y_idx, y_len, y_overlap = self._split_tiles(H)
        x_idx, x_len, x_overlap = self._split_tiles(W)

        # encode each tile
        rows: list[list[mx.array]] = []
        for i_pos, i_len in zip(y_idx, y_len):
            row: list[mx.array] = []
            for j_pos, j_len in zip(x_idx, x_len):
                tile = x_ndhwc[:, :, i_pos:i_pos + i_len, j_pos:j_pos + j_len, :]
                row.append(self._encode_moments_ndhwc(tile))
            rows.append(row)

        # latent-space overlap (in tokens)
        lat_y_overlap = [o // self.vae_ratio for o in y_overlap]
        lat_x_overlap = [o // self.vae_ratio for o in x_overlap]

        result_rows: list[mx.array] = []
        for i, row in enumerate(rows):
            result_row: list[mx.array] = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend_axis(rows[i - 1][j], tile, lat_y_overlap[i - 1], axis=2)
                if j > 0:
                    tile = self._blend_axis(row[j - 1], tile, lat_x_overlap[j - 1], axis=3)
                if i < len(rows) - 1:
                    tile = tile[:, :, : -lat_y_overlap[i], :, :]
                if j < len(row) - 1:
                    tile = tile[:, :, :, : -lat_x_overlap[j], :]
                result_row.append(tile)
            result_rows.append(mx.concatenate(result_row, axis=3))
        return mx.concatenate(result_rows, axis=2)

    def tiled_decode_ndhwc(self, z_ndhwc: mx.array) -> mx.array:
        """Overlapping spatial-tiled decode.

        ``z_ndhwc`` shape (B, T_lat, H_lat, W_lat, C_z). Output shape
        (B, T_out, H_out, W_out, C_out).
        """
        H_pixels = int(z_ndhwc.shape[2]) * self.vae_ratio
        W_pixels = int(z_ndhwc.shape[3]) * self.vae_ratio
        y_idx, y_len, y_overlap = self._split_tiles(H_pixels)
        x_idx, x_len, x_overlap = self._split_tiles(W_pixels)

        # decode each tile in row-major
        rows: list[list[mx.array]] = []
        for i_pos, i_len in zip(y_idx, y_len):
            zi = i_pos // self.vae_ratio
            zl = i_len // self.vae_ratio
            row: list[mx.array] = []
            for j_pos, j_len in zip(x_idx, x_len):
                zj = j_pos // self.vae_ratio
                zw = j_len // self.vae_ratio
                tile_z = z_ndhwc[:, :, zi:zi + zl, zj:zj + zw, :]
                row.append(self._decode_pixels_ndhwc(tile_z))
            rows.append(row)

        # blend + trim in pixel space (blend_extent = pixel overlap)
        result_rows: list[mx.array] = []
        for i, row in enumerate(rows):
            result_row: list[mx.array] = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend_axis(rows[i - 1][j], tile, y_overlap[i - 1], axis=2)
                if j > 0:
                    tile = self._blend_axis(row[j - 1], tile, x_overlap[j - 1], axis=3)
                if i < len(rows) - 1:
                    tile = tile[:, :, : -y_overlap[i], :, :]
                if j < len(row) - 1:
                    tile = tile[:, :, :, : -x_overlap[j], :]
                result_row.append(tile)
            result_rows.append(mx.concatenate(result_row, axis=3))
        return mx.concatenate(result_rows, axis=2)

    # ------------------------------------------------------------------
    # Phase 8.9-a: temporal chunk decode with overlap + cross-fade.
    # Ported from ComfyUI comfy/ldm/minimax/vae.py:426-651. NDHWC layout,
    # so the temporal axis is axis=1 (not axis=2 as in the torch NCDHW ref).
    # ------------------------------------------------------------------
    @staticmethod
    def _blend_axis1(a: "mx.array", b: "mx.array", blend_extent: int) -> "mx.array":
        """Linear cross-fade of ``a``'s tail into ``b``'s head along axis=1.

        Returns ``concat([blend, b_tail], axis=1)`` where ``blend`` has length
        ``blend_extent`` and ``b_tail = b[:, blend_extent:]``.
        """
        blend_extent = int(min(a.shape[1], b.shape[1], blend_extent))
        if blend_extent <= 0:
            return b
        pos = mx.arange(blend_extent, dtype=b.dtype)
        w_b = (pos / blend_extent).reshape(1, blend_extent, 1, 1, 1)
        w_a = 1.0 - w_b
        a_tail = a[:, -blend_extent:, :, :, :]
        b_head = b[:, :blend_extent, :, :, :]
        blended = a_tail * w_a + b_head * w_b
        if blend_extent < b.shape[1]:
            return mx.concatenate([blended, b[:, blend_extent:, :, :, :]], axis=1)
        return blended

    def _decode_temporal_pad_frames(self, z_len: int, pad_tokens: int) -> int:
        if pad_tokens <= 0:
            return 0
        intra_tail = self.clip_length % self.vae_ratio_t
        if intra_tail == 0:
            return pad_tokens * self.vae_ratio_t
        z_len_before_pad = z_len - pad_tokens
        return sum(
            intra_tail if (z_len_before_pad + k) % self.tokens_chunk_size == 0
            else self.vae_ratio_t
            for k in range(pad_tokens)
        )

    def _decode_temporal_frame_plan(self, z_len: int, num_chunks: int, pad_tokens: int) -> int:
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1
        total_frames = 0
        final_overlap_frames = 0
        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_token_len = max(0, min(t_end_idx, z_len) - min(t_start_idx, z_len))
            clip_frame_len = clip_token_len * self.vae_ratio_t
            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_frame_len)
                chunk_frames = max(0, f_end_idx - f_start_idx - self.frame_pre_padding)
                if j == 0:
                    total_frames += chunk_frames
                else:
                    final_overlap_frames = chunk_frames
        total_frames += final_overlap_frames
        return total_frames - self._decode_temporal_pad_frames(z_len, pad_tokens)

    def decode_temporal_ndhwc(self, z: "mx.array") -> "mx.array":
        """NDHWC decode_temporal. ``z`` is (B, T_lat, H_lat, W_lat, C_z)."""
        chunk_dec = self.tokens_chunk_size * self.vae_ratio_t
        split_count = int(self.token_drop > 0) + 1

        T_lat = z.shape[1]
        pseudo_total_tokens = T_lat + self.token_drop

        pad_tokens = 0
        remainder = pseudo_total_tokens % self.tokens_chunk_size
        if remainder != 0:
            pad_tokens = self.tokens_chunk_size - remainder
            pseudo_total_tokens += pad_tokens

        num_chunks = pseudo_total_tokens // self.tokens_chunk_size - int(self.token_drop > 0)
        if num_chunks < 1:
            pad_tokens += self.tokens_chunk_size
            num_chunks += 1

        if pad_tokens > 0:
            pad_z = mx.broadcast_to(
                z[:, -1:, :, :, :],
                (z.shape[0], pad_tokens, z.shape[2], z.shape[3], z.shape[4]),
            )
            z = mx.concatenate([z, pad_z], axis=1)

        output_frames = self._decode_temporal_frame_plan(z.shape[1], num_chunks, pad_tokens)

        parts = []          # list of (write_pos, part) tuples
        dec_overlap = None
        write_pos = 0
        first_dtype = None
        first_shape_tail = None  # (H, W, C)

        for i in range(num_chunks):
            t_start_idx = i * self.tokens_chunk_size
            t_end_idx = t_start_idx + self.tokens_chunk_size + self.token_overlap
            clip_z = z[:, t_start_idx:t_end_idx, :, :, :]

            clip_dec = self._adaptive_decode_ndhwc(clip_z)
            if first_dtype is None:
                first_dtype = clip_dec.dtype
                first_shape_tail = clip_dec.shape[2:]

            for j in range(split_count):
                f_start_idx = j * chunk_dec
                f_end_idx = min(f_start_idx + chunk_dec, clip_dec.shape[1])
                clip_dec_chunk = clip_dec[:, f_start_idx:f_end_idx, :, :, :]
                clip_dec_chunk = clip_dec_chunk[:, self.frame_pre_padding:, :, :, :]

                if j == 0:
                    if dec_overlap is not None:
                        clip_dec_chunk = self._blend_axis1(
                            dec_overlap, clip_dec_chunk, self.frame_overlap
                        )
                        dec_overlap = None
                    part_frames = clip_dec_chunk.shape[1]
                    copy_frames = min(part_frames, max(0, output_frames - write_pos))
                    if copy_frames > 0:
                        parts.append(clip_dec_chunk[:, :copy_frames, :, :, :])
                        write_pos += copy_frames
                else:
                    dec_overlap = clip_dec_chunk

            if i == num_chunks - 1 and dec_overlap is not None:
                part_frames = dec_overlap.shape[1]
                copy_frames = min(part_frames, max(0, output_frames - write_pos))
                if copy_frames > 0:
                    parts.append(dec_overlap[:, :copy_frames, :, :, :])
                    write_pos += copy_frames
                dec_overlap = None

        if not parts:
            # Should not happen for T_lat >= 1, but keep a safe empty tensor.
            return mx.zeros(
                (z.shape[0], 0) + first_shape_tail, dtype=first_dtype or z.dtype
            )
        return mx.concatenate(parts, axis=1)

    def encode(self, x: mx.array) -> mx.array:
        """Encode NCDHW pixels in [-1, 1] to normalized latents (mean only).

        Accepts a 4-D image ``(B, C, H, W)`` (auto-unsqueezed to 1 frame) or a
        5-D video ``(B, C, T, H, W)`` where ``T`` divides ``clip_length``.
        Only the fast single-clip path is exercised in Phase 2.
        """
        if x.ndim == 4:
            x = x[:, :, None, :, :]

        # Pixel normalize:  (x+1)/2 then imagenet normalize.
        x = self._nchw_to_ndhwc(x)
        x = (x + 1.0) * 0.5
        x = (x - self.pixel_mean.astype(x.dtype)) / self.pixel_std.astype(x.dtype)

        if x.shape[1] == 1:
            moments = self._adaptive_encode_ndhwc(x)
            moments = moments[:, -1:, :, :, :]
        else:
            # Simple temporal-chunked path: split into non-overlapping clips.
            if x.shape[1] % self.clip_length != 0:
                pad = (-x.shape[1]) % self.clip_length
                tail = mx.broadcast_to(
                    x[:, -1:, :, :, :],
                    (x.shape[0], pad, x.shape[2], x.shape[3], x.shape[4]),
                )
                x = mx.concatenate([x, tail], axis=1)
            num_chunks = x.shape[1] // self.clip_length
            outs = []
            for i in range(num_chunks):
                clip = x[:, i * self.clip_length:(i + 1) * self.clip_length]
                outs.append(self._adaptive_encode_ndhwc(clip))
            moments = mx.concatenate(outs, axis=1)
            if self.token_drop > 0:
                moments = moments[:, :-self.token_drop]

        # moments -> mean chunk (first half of channels)
        moments = moments.astype(mx.float32)
        C = moments.shape[-1]
        mean = moments[:, :, :, :, : C // 2]

        latents_mean = self.latents_mean.reshape(1, 1, 1, 1, -1).astype(mean.dtype)
        latents_std = self.latents_std.reshape(1, 1, 1, 1, -1).astype(mean.dtype)
        z_ndhwc = (mean - latents_mean) / latents_std
        return self._ndhwc_to_nchw(z_ndhwc)

    # ------------------------------------------------------------------
    # Phase 8.7 patch-boundary deblock (post-decoder low-pass across seams)
    # ------------------------------------------------------------------
    def _deblock_patches(self, x_ncdhw: mx.array) -> mx.array:
        """Blend a narrow band around every ``vae_ratio``-pixel boundary in H, W.

        For each boundary at coordinate ``k * patch`` (k > 0) and each offset
        ``o`` in 1..blend_width, we average the pixel at ``k*patch - o``
        with its mirror at ``k*patch + o - 1``, using triangular weights that
        max at ``o=1`` and taper to zero at ``o=blend_width``. This kills the
        16-pixel crosshatch texture without touching interior pixels.
        """
        patch = self.vae_ratio
        bw = self.deblock_blend_width
        alpha = self.deblock_alpha
        if not self.deblock_patches or bw <= 0 or alpha <= 0.0:
            return x_ncdhw
        # NCDHW layout; work on H (axis=3) and W (axis=4) in place with slices
        B, C, T, H, W = x_ncdhw.shape
        y = x_ncdhw
        # Column boundaries (blend along W axis)
        for k in range(1, W // patch):
            x0 = k * patch
            for off in range(1, bw + 1):
                w = alpha * (1.0 - (off - 1) / bw)
                lx = x0 - off
                rx = x0 + off - 1
                if lx < 0 or rx >= W:
                    continue
                left = y[:, :, :, :, lx:lx + 1]
                right = y[:, :, :, :, rx:rx + 1]
                new_left = left * (1.0 - w) + right * w
                new_right = right * (1.0 - w) + left * w
                y = mx.concatenate([
                    y[:, :, :, :, :lx], new_left,
                    y[:, :, :, :, lx + 1:rx], new_right,
                    y[:, :, :, :, rx + 1:],
                ], axis=-1)
        # Row boundaries (blend along H axis)
        for k in range(1, H // patch):
            y0 = k * patch
            for off in range(1, bw + 1):
                w = alpha * (1.0 - (off - 1) / bw)
                ly = y0 - off
                ry = y0 + off - 1
                if ly < 0 or ry >= H:
                    continue
                top = y[:, :, :, ly:ly + 1, :]
                bot = y[:, :, :, ry:ry + 1, :]
                new_top = top * (1.0 - w) + bot * w
                new_bot = bot * (1.0 - w) + top * w
                y = mx.concatenate([
                    y[:, :, :, :ly, :], new_top,
                    y[:, :, :, ly + 1:ry, :], new_bot,
                    y[:, :, :, ry + 1:, :],
                ], axis=-2)
        return y

    def decode(self, z: mx.array) -> mx.array:
        """Decode normalized latents (NCDHW) to pixels in [-1, 1] (NCDHW)."""
        if z.ndim == 4:
            z = z[:, :, None, :, :]

        z = self._nchw_to_ndhwc(z)
        latents_mean = self.latents_mean.reshape(1, 1, 1, 1, -1).astype(z.dtype)
        latents_std = self.latents_std.reshape(1, 1, 1, 1, -1).astype(z.dtype)
        z = z * latents_std + latents_mean

        if z.shape[1] == 1:
            dec = self._adaptive_decode_ndhwc(z)
            dec = dec[:, -1:, :, :, :]
        else:
            # Phase 8.9-a: proper temporal chunk decode with overlap +
            # cross-fade (matches ComfyUI decode_temporal).
            dec = self.decode_temporal_ndhwc(z)

        dec = dec.astype(mx.float32)
        dec = dec * self.pixel_std.astype(dec.dtype) + self.pixel_mean.astype(dec.dtype)
        dec = mx.clip(dec, 0.0, 1.0) * 2.0 - 1.0
        dec_ncdhw = self._ndhwc_to_nchw(dec)
        # Phase 8.7: kill the 16-px patch grid before returning
        dec_ncdhw = self._deblock_patches(dec_ncdhw)
        return dec_ncdhw


__all__ = [
    "CausalConv3d",
    "TemporalIsolatedGroupNorm",
    "Downsample3D",
    "ResnetBlock3D",
    "EncoderFCN3D",
    "PointwiseConv3d",
    "RMSNormAffine",
    "LayerNormAffine",
    "RotaryEmbeddingND",
    "GatedFeedForward",
    "ViTAttention",
    "TransformerBlock",
    "ViT3DDecoder",
    "MiniMaxH3VideoVAE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LATENTS_MEAN",
    "LATENTS_STD",
]
