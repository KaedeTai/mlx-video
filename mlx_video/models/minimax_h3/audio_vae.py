"""MiniMax H3 Audio VAE — MLX port.

DAC-lineage waveform encoder + BigVGAN vocoder decoder, 32 kHz stereo,
40 latent frames per second.

Reference: /tmp/h3_recon/ComfyUI/comfy/ldm/minimax/audio_vae.py (443 LOC).

Tensor layout
-------------
MLX Conv1d/ConvTranspose1d work in **NLC** (batch, length, channels). PyTorch
audio code — and the reference module — works in **NCL**. The public
`encode`/`decode` functions accept and return NCL tensors so their signature
matches the reference; every internal op runs in NLC and we `swapaxes(1, 2)`
at the boundary.

Weight-norm parametrizations (`weight_g` / `weight_v`) are folded into plain
`weight` tensors at load time (see `convert.py`), so this module stores plain
`weight`/`bias` on every convolution.
"""

from __future__ import annotations

import math
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Latents normalization constants (from Ref2VA/audio_vae/config.json)
# ---------------------------------------------------------------------------

LATENTS_MEAN = [
    -0.020211687488382354,  0.3876466479950502,  -0.04398279799186767,
    -0.28591514936373,       0.08179686214561671, -0.35782641352446604,
     0.040623809960919084,  -0.01552534501956604, -0.223362481667332,
     0.1821006842509091,     0.2941778783780663,  -0.07901167601970885,
    -0.056815072777201,     -0.3699028221860095,  -0.31616315591624855,
     0.5905951377425391,    -0.052139568068853864, 0.013673160263486295,
    -0.03691647864630577,    0.09732660653298163, -0.3394662328788498,
    -0.30685677538541667,   -0.24504598907458763, -0.034698524462007344,
     0.02868032184767538,   -0.21217779266454084, -0.1678263169941987,
     0.3221287889040614,    -0.1223055851554907,   0.4356604928128464,
    -0.0502599202236253,     0.3979258376211797,
]

LATENTS_STD = [
    1.6895524230479284, 2.76263727217653,   1.7945344281264435,
    1.6801681847309828, 1.6390226546605453, 2.7788298348882177,
    1.7659090095747236, 1.6199757612137327, 2.6336525640336896,
    1.8539356672817833, 2.5056497896915633, 1.811019237886178,
    1.9579657790720237, 1.6685498243529284, 1.4922469314453364,
    3.298670198067373,  1.9491804496832168, 1.8720003270431442,
    1.8334080103291832, 1.6488070416529093, 1.6176957696319716,
    1.9131449234774398, 1.5695245398428617, 1.6943659940415912,
    1.8318420762504692, 1.5540637421583379, 1.9344930328968526,
    1.599198216109855,  1.718045989838149,  1.6307219190837705,
    1.8661226051202384, 1.5613768203168363,
]


# ---------------------------------------------------------------------------
# Deterministic Kaiser-windowed sinc filter (matches the reference)
# ---------------------------------------------------------------------------


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> mx.array:
    """Returns a `(1, 1, kernel_size)` low-pass filter identical to the ref."""
    import numpy as np

    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0

    # np's kaiser_window matches torch.kaiser_window(periodic=False)
    window = np.kaiser(kernel_size, beta).astype(np.float32)

    if even:
        time = np.arange(-half_size, half_size, dtype=np.float32) + 0.5
    else:
        time = (np.arange(kernel_size) - half_size).astype(np.float32)

    if cutoff == 0:
        filter_ = np.zeros_like(time)
    else:
        arg = 2 * cutoff * time
        # np.sinc(x) := sin(pi*x)/(pi*x) — same as torch.sinc.
        filter_ = 2 * cutoff * window * np.sinc(arg)
        filter_ = filter_ / filter_.sum()

    return mx.array(filter_.reshape(1, 1, kernel_size))


# ---------------------------------------------------------------------------
# Snake activations
# ---------------------------------------------------------------------------


class Snake1d(nn.Module):
    """DAC-style Snake: `x + (1/alpha) * sin(alpha * x)^2`, per-channel alpha."""

    def __init__(self, channels: int):
        super().__init__()
        # Kept in the checkpoint's (1, C, 1) NCL layout so state-dict load is
        # a straight assign; broadcast in NLC by reshaping on the fly.
        self.alpha = mx.ones((1, channels, 1))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, L, C).  alpha stored as (1, C, 1) → view as (1, 1, C).
        a = self.alpha.reshape(1, 1, -1)
        ax = a * x
        return x + mx.sin(ax) * mx.sin(ax) / (a + 1e-9)


class SnakeBeta(nn.Module):
    """BigVGAN Snake with separate alpha (freq) and beta (magnitude), log-scaled."""

    def __init__(self, channels: int):
        super().__init__()
        # alpha_logscale=True in the 32 kHz preset → initialize at zero (== log 1).
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, L, C). alpha/beta: (C,) → broadcast on N and L via reshape.
        a = mx.exp(self.alpha).reshape(1, 1, -1)
        b = mx.exp(self.beta).reshape(1, 1, -1)
        ax = a * x
        return x + mx.sin(ax) * mx.sin(ax) / (b + 1e-9)


# ---------------------------------------------------------------------------
# Anti-aliased (StyleGAN3-style) up/downsamplers and Activation1d wrapper
# ---------------------------------------------------------------------------


def _edge_pad_l(x: mx.array, pad_left: int, pad_right: int) -> mx.array:
    """PyTorch 'replicate' pad on the L axis of an NLC tensor."""
    if pad_left == 0 and pad_right == 0:
        return x
    return mx.pad(x, [(0, 0), (pad_left, pad_right), (0, 0)], mode="edge")


class UpSample1d(nn.Module):
    """Anti-aliased 2x upsampler (replicate-pad -> grouped conv_transpose -> crop)."""

    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (kernel_size - self.stride + 1) // 2
        # Buffer stored in checkpoint layout `(1, 1, K)`; kept as-is for load parity.
        self.filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, L, C)
        C = x.shape[-1]
        x = _edge_pad_l(x, self.pad, self.pad)
        # filter (1, 1, K) → expand to (C, 1, K) then permute to MLX (C, K, 1).
        w = mx.broadcast_to(self.filter, (C, 1, self.filter.shape[-1]))
        w = mx.swapaxes(w, 1, 2)  # (C, K, 1)
        x = mx.conv_transpose1d(x, w, stride=self.stride, groups=C) * self.ratio
        # Crop [:, pad_left : -pad_right, :] along L
        return x[:, self.pad_left : -self.pad_right, :]


class LowPassFilter1d(nn.Module):
    def __init__(self, cutoff: float = 0.5, half_width: float = 0.6, stride: int = 1, kernel_size: int = 12):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad_left = kernel_size // 2 - int(kernel_size % 2 == 0)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        C = x.shape[-1]
        x = _edge_pad_l(x, self.pad_left, self.pad_right)
        w = mx.broadcast_to(self.filter, (C, 1, self.filter.shape[-1]))
        w = mx.swapaxes(w, 1, 2)
        return mx.conv1d(x, w, stride=self.stride, groups=C)


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=kernel_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.lowpass(x)


class Activation1d(nn.Module):
    """upsample x2 → pointwise activation → downsample x2 (anti-aliased)."""

    def __init__(self, activation: nn.Module, up_ratio: int = 2, down_ratio: int = 2,
                 up_kernel_size: int = 12, down_kernel_size: int = 12):
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)


# ---------------------------------------------------------------------------
# Plain 1D conv modules (weight-norm already folded at load time)
# ---------------------------------------------------------------------------


class WNConv1d(nn.Module):
    """A plain Conv1d whose weight was originally weight-normed (folded on load).

    Attribute layout matches MLX's Conv1d: `weight` shape `(O, K, I)`, `bias` `(O,)`.
    Bias is optional.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None  # not registered as parameter (assigned attribute)

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv1d(x, self.weight, stride=self.stride, padding=self.padding, dilation=self.dilation)
        if self.bias is not None:
            y = y + self.bias
        return y


class WNConvTranspose1d(nn.Module):
    """Plain ConvTranspose1d, weight-norm folded on load."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.stride = stride
        self.padding = padding
        # MLX layout (C_out, K, C_in)
        self.weight = mx.zeros((out_channels, kernel_size, in_channels))
        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.conv_transpose1d(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            y = y + self.bias
        return y


# ---------------------------------------------------------------------------
# DAC encoder
# ---------------------------------------------------------------------------


class ResidualUnit(nn.Module):
    """Snake / dilated k=7 conv / Snake / k=1 conv, with center-cropped skip."""

    def __init__(self, dim: int, dilation: int):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = [
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        y = x
        for m in self.block:
            y = m(y)
        # Center-crop skip when the dilated conv shrunk L (pad=(6*dil)//2 may not
        # match the required amount).
        diff = x.shape[1] - y.shape[1]
        if diff > 0:
            crop = diff // 2
            x = x[:, crop : x.shape[1] - crop, :]
        return x + y


class EncoderBlock(nn.Module):
    """3× ResidualUnit(dilations 1/3/9) then Snake+strided conv, doubling channels."""

    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = [
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(dim // 2, dim, kernel_size=2 * stride, stride=stride,
                     padding=math.ceil(stride / 2)),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for m in self.block:
            x = m(x)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model: int = 64, strides: Sequence[int] = (2, 4, 4, 5, 5), d_latent: int = 2048):
        super().__init__()
        blocks = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            blocks.append(EncoderBlock(d_model, stride=stride))
        blocks.append(Snake1d(d_model))
        blocks.append(WNConv1d(d_model, d_latent, kernel_size=3, padding=1))
        # NLE keys `encoder.block.0..7`
        self.block = blocks

    def __call__(self, x: mx.array) -> mx.array:
        for m in self.block:
            x = m(x)
        return x


# ---------------------------------------------------------------------------
# AttnProjection posterior head
# ---------------------------------------------------------------------------


class GeGluMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.w0 = nn.Linear(in_features, hidden_features)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(hidden_features, in_features)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        # GELU tanh-approx: same as torch nn.GELU(approximate="tanh")
        gate = nn.gelu_approx(self.w0(x))
        return self.w2(gate * self.w1(x))


def _adaptive_avg_pool_last(x: mx.array, out_size: int) -> mx.array:
    """Adaptive average pool over the last dim of x, matching PyTorch semantics.

    PyTorch computes for each output index i:
        start = floor(i * L / out_size)
        end   = ceil((i + 1) * L / out_size)
    and averages `x[..., start:end]`.
    """
    L = x.shape[-1]
    if L == out_size:
        return x
    # Precompute segment boundaries (constant, tiny loop over out_size).
    outs = []
    for i in range(out_size):
        start = (i * L) // out_size
        end = -(-(i + 1) * L // out_size)  # ceil-div
        outs.append(mx.mean(x[..., start:end], axis=-1, keepdims=True))
    return mx.concatenate(outs, axis=-1)


class CausalAttention(nn.Module):
    """Causal SDPA + mean-over-heads + adaptive-pool projection (in_dim > out_dim path)."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int):
        super().__init__()
        assert in_dim > out_dim, "H3 audio VAE always uses the in_dim > out_dim branch"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # qkv has no bias parameter — bias is stitched together at forward time
        # from q_bias, zero_k_bias, v_bias so k has a hard zero bias.
        self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
        self.q_bias = mx.zeros((in_dim,))
        self.v_bias = mx.zeros((in_dim,))
        # zero_k_bias is a buffer in the reference (present in the checkpoint but
        # always zero). Keep it as a Module attribute so state-dict load populates it.
        self.zero_k_bias = mx.zeros((in_dim,))

        self.proj = nn.Linear(out_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        qkv_bias = mx.concatenate([self.q_bias, self.zero_k_bias, self.v_bias], axis=0)
        qkv = self.qkv(x) + qkv_bias  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))  # (3, B, H, N, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # SDPA with causal mask.
        attn_out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask="causal",
        )  # (B, H, N, hd)

        # Mean over head dim → (B, N, hd), then adaptive-pool head_dim → out_dim.
        pooled = mx.mean(attn_out, axis=1)  # (B, N, hd)
        if self.head_dim != self.out_dim:
            pooled = _adaptive_avg_pool_last(pooled, self.out_dim)
        return self.proj(pooled)


class AttnProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, mlp_ratio: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        hidden = int(out_dim * mlp_ratio)
        self.mlp = GeGluMlp(in_features=out_dim, hidden_features=hidden)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, N, in_dim)
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


# ---------------------------------------------------------------------------
# BigVGAN decoder
# ---------------------------------------------------------------------------


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class AMPBlock1(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = [
            WNConv1d(channels, channels, kernel_size, stride=1, dilation=d,
                     padding=_get_padding(kernel_size, d))
            for d in dilation
        ]
        self.convs2 = [
            WNConv1d(channels, channels, kernel_size, stride=1, dilation=1,
                     padding=_get_padding(kernel_size, 1))
            for _ in range(len(dilation))
        ]
        self.num_layers = len(self.convs1) + len(self.convs2)
        self.activations = [
            Activation1d(activation=SnakeBeta(channels)) for _ in range(self.num_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x


class BigVGAN(nn.Module):
    """32 kHz BigVGAN preset used by MiniMax H3.

    `use_bias_at_final=False`, `use_tanh_at_final=False` → clamp output to [-1, 1].
    """

    def __init__(
        self,
        num_mels: int = 2048,
        upsample_initial_channel: int = 1024,
        upsample_rates=(5, 5, 2, 2, 2, 2, 2),
        upsample_kernel_sizes=(9, 9, 4, 4, 4, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = WNConv1d(num_mels, upsample_initial_channel, 7, stride=1, padding=3)

        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append([
                WNConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k, stride=u, padding=(k - u) // 2,
                )
            ])

        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(ch, k, d))

        self.activation_post = Activation1d(activation=SnakeBeta(ch))
        self.conv_post = WNConv1d(ch, 1, kernel_size=7, stride=1, padding=3, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, L, num_mels) NLC
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            for u in self.ups[i]:
                x = u(x)
            xs = None
            for j in range(self.num_kernels):
                block = self.resblocks[i * self.num_kernels + j]
                xs = block(x) if xs is None else xs + block(x)
            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        return mx.clip(x, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Top-level VAE
# ---------------------------------------------------------------------------


class MiniMaxH3AudioVAE(nn.Module):
    """MiniMax H3 stereo audio VAE at 32 kHz.

    Latents shape: `(B, 32, 2, T)` — 32 channels × 2 stereo × T frames at 40 fps.
    Each stereo channel is encoded / decoded independently through a mono path.
    """

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: Sequence[int] = (2, 4, 4, 5, 5),
        latent_dim: int = 2048,
        decoder_dim: int = 1024,
        vae_latent_channels: int = 32,
    ):
        super().__init__()
        self.sample_rate = 32000

        self.hop_length = 1
        for r in encoder_rates:
            self.hop_length *= r  # 800
        self.samples_per_latent = self.hop_length
        self.latents_per_second = self.sample_rate // self.hop_length  # 40
        self.output_sample_rate = self.sample_rate

        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        self.pre_block = AttnProjection(latent_dim, vae_latent_channels, num_heads=8)

        # 1×1 convs — no weight-norm in the checkpoint.
        self.mean_proj = WNConv1d(vae_latent_channels, vae_latent_channels, 1)
        self.logs_proj = WNConv1d(vae_latent_channels, vae_latent_channels, 1)
        self.dec_in_proj = WNConv1d(vae_latent_channels, latent_dim, 1)

        self.decoder = BigVGAN(
            num_mels=latent_dim, upsample_initial_channel=decoder_dim
        )

        # Latents_mean / std live in config.json, not the safetensors.
        self.latents_mean = mx.array(LATENTS_MEAN[:vae_latent_channels], dtype=mx.float32)
        self.latents_std = mx.array(LATENTS_STD[:vae_latent_channels], dtype=mx.float32)

    # -------- helpers --------

    @staticmethod
    def _ncl_to_nlc(x: mx.array) -> mx.array:
        return mx.swapaxes(x, 1, 2)

    @staticmethod
    def _nlc_to_ncl(x: mx.array) -> mx.array:
        return mx.swapaxes(x, 1, 2)

    # -------- public API --------

    def encode(self, waveform: mx.array) -> mx.array:
        """Encode stereo waveform `(B, 2, L)` in [-1, 1] to normalized latents `(B, 32, 2, T)`."""
        b, s, length = waveform.shape
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        if right_pad > 0:
            waveform = mx.pad(waveform, [(0, 0), (0, 0), (0, right_pad)])

        # (b*s, 1, L) NCL → (b*s, L, 1) NLC
        x = waveform.reshape(b * s, 1, -1)
        x = self._ncl_to_nlc(x)

        x = self.encoder(x)                        # (b*s, T, latent_dim=2048)
        x = self.pre_block(x)                      # (b*s, T, 32)  attention head
        z = self.mean_proj(x)                      # (b*s, T, 32)  1x1 conv, NLC-native

        # normalize per channel
        mean = self.latents_mean.astype(z.dtype).reshape(1, 1, -1)
        std = self.latents_std.astype(z.dtype).reshape(1, 1, -1)
        z = (z - mean) / std

        # (b*s, T, 32) → (b, s, T, 32) → (b, 32, s, T)
        T = z.shape[1]
        z = z.reshape(b, s, T, -1)                 # (b, s, T, 32)
        z = mx.transpose(z, (0, 3, 1, 2))          # (b, 32, s, T)
        return z

    def decode(self, z: mx.array) -> mx.array:
        """Decode normalized latents `(B, 32, 2, T)` to stereo waveform `(B, 2, L)`."""
        b, c, s, t = z.shape
        # (b, c, s, t) → (b, s, c, t) → (b*s, c, t) NCL
        z = mx.transpose(z, (0, 2, 1, 3)).reshape(b * s, c, t)
        z = self._ncl_to_nlc(z)                    # (b*s, t, c)  NLC

        mean = self.latents_mean.astype(z.dtype).reshape(1, 1, -1)
        std = self.latents_std.astype(z.dtype).reshape(1, 1, -1)
        z = z * std + mean

        x = self.dec_in_proj(z)                    # (b*s, t, latent_dim=2048)
        x = self.decoder(x)                        # (b*s, L, 1)  clamped
        L = x.shape[1]
        x = x.reshape(b, s, L)                     # (b, 2, L)
        return x
