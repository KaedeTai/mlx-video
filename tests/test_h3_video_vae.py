"""Numerical parity check: MLX MiniMaxH3VideoVAE vs the PyTorch reference.

We compare the CNN encoder path (input -> quant_conv -> mean chunk) and the
ViT decoder path (post_quant_conv -> decoder) separately so that regressions
localize to one half of the model. The reference is
``~/models/MiniMax-H3-raw/Ref2VA/video_vae`` loaded via ``diffusers`` (uses
torch SDPA, no CUDA / flash-attn dependency).

Ports are on CPU because the model is 2.6 B params and Mac unified memory
requires PyTorch on MPS to be careful, whereas CPU + fp32 gives a rock-solid
oracle. MLX runs on Metal by default.

Requires the converted safetensors to exist at
``~/mlx-video/mlx-models/MiniMaxH3-VideoVAE-MLX-bf16/model.safetensors``.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest


# --- test config --------------------------------------------------------

MLX_CKPT = Path("~/mlx-video/mlx-models/MiniMaxH3-VideoVAE-MLX-bf16/model.safetensors").expanduser()
PT_ROOT = Path("~/models/MiniMax-H3-raw/Ref2VA/video_vae").expanduser()

# Small enough to make CPU forward tractable (a full 384x384 pass on CPU is
# a couple of minutes). 64x64 → 4x4 latent still hits every code path.
SPATIAL = 64
FRAMES_ENC = 1  # single-frame path uses the "if T == 1" shortcut


def _psnr(a: np.ndarray, b: np.ndarray, peak: float = 2.0) -> float:
    """PSNR in dB using ``peak`` as the max value range (pixels are [-1,1] → peak=2)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return math.inf
    return 20.0 * math.log10(peak / math.sqrt(mse))


def _psnr_generic(a: np.ndarray, b: np.ndarray) -> float:
    """Peak-relative PSNR: peak = max(|a|,|b|), avoids underflow on latent-scale values."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    peak = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-6) * 2.0
    return _psnr(a, b, peak=peak)


def _skip_if_missing():
    if not MLX_CKPT.is_file():
        pytest.skip(f"missing MLX checkpoint: {MLX_CKPT} (run convert.py first)")
    if not (PT_ROOT / "config.json").is_file():
        pytest.skip(f"missing PyTorch VAE at {PT_ROOT}")


# --- fixtures -----------------------------------------------------------


@pytest.fixture(scope="module")
def mlx_vae():
    _skip_if_missing()
    import mlx.core as mx
    from mlx_video.models.minimax_h3.video_vae import MiniMaxH3VideoVAE
    m = MiniMaxH3VideoVAE()
    ck = mx.load(str(MLX_CKPT))
    m.load_weights(list(ck.items()), strict=False)
    return m


@pytest.fixture(scope="module")
def pt_vae():
    _skip_if_missing()
    sys.path.insert(0, str(PT_ROOT.parent))
    from video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE as PTVAE
    import torch
    m = PTVAE.from_pretrained(str(PT_ROOT))
    m.eval()
    m.to(torch.float32)
    return m


# --- tests --------------------------------------------------------------


def test_encoder_parity(pt_vae, mlx_vae):
    """CNN encoder + quant_conv + mean-chunk should match to > 30 dB."""
    import mlx.core as mx
    import torch

    np.random.seed(0)
    x_np = np.random.randn(1, 3, FRAMES_ENC, SPATIAL, SPATIAL).astype(np.float32) * 0.3

    # PT: uses full pipeline. Their `encode()` returns 48-ch moments (no norm).
    with torch.no_grad():
        pt_moments = pt_vae.model.quant_conv(pt_vae.model.encoder(torch.from_numpy(x_np)))
    pt_mean = pt_moments.numpy()[:, :24]  # first-half chunk

    # MLX: same raw path. We call encoder → quant_conv → mean chunk directly.
    x_mlx = mx.array(x_np).transpose(0, 2, 3, 4, 1)  # NCDHW -> NDHWC
    mlx_moments = mlx_vae.quant_conv(mlx_vae.encoder(x_mlx))
    mlx_mean = np.array(mlx_moments[:, :, :, :, :24].astype(mx.float32))
    # NDHWC -> NCDHW to compare
    mlx_mean = mlx_mean.transpose(0, 4, 1, 2, 3)

    psnr = _psnr_generic(pt_mean, mlx_mean)
    print(f"[encoder] PSNR = {psnr:.2f} dB, "
          f"pt range=[{pt_mean.min():.3f}, {pt_mean.max():.3f}], "
          f"mlx range=[{mlx_mean.min():.3f}, {mlx_mean.max():.3f}]")
    assert psnr > 30.0, f"encoder PSNR {psnr:.2f} dB < 30 dB"


def test_decoder_parity(pt_vae, mlx_vae):
    """ViT decoder + post_quant_conv should match to > 30 dB."""
    import mlx.core as mx
    import torch

    # Small latent: 1 frame -> 1 token temporal, 4x4 spatial.
    np.random.seed(0)
    z_np = (np.random.randn(1, 24, 1, 4, 4).astype(np.float32) * 0.5)

    with torch.no_grad():
        pt_dec = pt_vae.model.decoder(pt_vae.model.post_quant_conv(torch.from_numpy(z_np)))
    pt_dec_np = pt_dec.numpy()

    z_mlx = mx.array(z_np).transpose(0, 2, 3, 4, 1)  # NCDHW -> NDHWC
    mlx_dec = mlx_vae.decoder(mlx_vae.post_quant_conv(z_mlx))
    mlx_dec_np = np.array(mlx_dec.astype(mx.float32)).transpose(0, 4, 1, 2, 3)

    psnr = _psnr_generic(pt_dec_np, mlx_dec_np)
    print(f"[decoder] PSNR = {psnr:.2f} dB, "
          f"pt range=[{pt_dec_np.min():.3f}, {pt_dec_np.max():.3f}], "
          f"mlx range=[{mlx_dec_np.min():.3f}, {mlx_dec_np.max():.3f}]")
    assert psnr > 30.0, f"decoder PSNR {psnr:.2f} dB < 30 dB"


def test_full_round_trip_on_face(pt_vae, mlx_vae, tmp_path):
    """Encode -> decode a real face image; compare MLX vs PyTorch reference.

    This VAE is designed for 17-frame clips (patch_size_t=4), so a single-frame
    image reconstructs at ~15 dB even in the PyTorch reference. What matters
    for the port is that MLX matches the reference within a fraction of a dB;
    absolute PSNR floor for image reconstruction is documented but not asserted.
    """
    import mlx.core as mx
    import torch

    face_path = Path("~/movie/wang_wenchin/faces/0100.jpg").expanduser()
    if not face_path.is_file():
        pytest.skip(f"missing test image {face_path}")

    from PIL import Image
    img = Image.open(face_path).convert("RGB").resize((256, 256))
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
    x_np = arr.transpose(2, 0, 1)[None, :, None, :, :]  # (1, 3, 1, 256, 256)

    # MLX round-trip
    x_mlx = mx.array(x_np)
    t0 = time.time()
    z_mlx = mlx_vae.encode(x_mlx)
    mx.eval(z_mlx)
    t_enc = time.time() - t0
    t0 = time.time()
    dec_mlx = mlx_vae.decode(z_mlx)
    mx.eval(dec_mlx)
    t_dec = time.time() - t0
    dec_mlx_np = np.array(dec_mlx.astype(mx.float32))

    # PyTorch round-trip (through the same pixel-normalize + latent-normalize
    # pipeline we use in MLX). We mimic the ComfyUI reference wrapper because
    # the raw diffusers ``encode()`` returns un-normalized moments.
    from mlx_video.models.minimax_h3.video_vae import (
        IMAGENET_MEAN, IMAGENET_STD, LATENTS_MEAN, LATENTS_STD,
    )
    mean_img = np.array(IMAGENET_MEAN).reshape(1, 3, 1, 1, 1)
    std_img = np.array(IMAGENET_STD).reshape(1, 3, 1, 1, 1)
    lat_m = np.array(LATENTS_MEAN).reshape(1, 24, 1, 1, 1)
    lat_s = np.array(LATENTS_STD).reshape(1, 24, 1, 1, 1)

    x_norm = ((x_np + 1) * 0.5 - mean_img) / std_img
    with torch.no_grad():
        moments = pt_vae.model.quant_conv(
            pt_vae.model.encoder(torch.from_numpy(x_norm.astype(np.float32)))
        )
        pt_mean = moments.numpy()[:, :24]
        # Normalize -> denormalize (round-trip through the same latent scaling)
        z_pt_norm = (pt_mean - lat_m) / lat_s
        pt_z_denorm = z_pt_norm * lat_s + lat_m
        pt_dec = pt_vae.model.decoder(
            pt_vae.model.post_quant_conv(torch.from_numpy(pt_z_denorm.astype(np.float32)))
        )
    pt_dec_np = pt_dec.numpy() * std_img + mean_img
    pt_dec_np = np.clip(pt_dec_np, 0, 1) * 2 - 1
    pt_dec_last = pt_dec_np[:, :, -1:]

    mlx_vs_pt_psnr = _psnr_generic(pt_dec_last, dec_mlx_np)
    mlx_vs_input_psnr = _psnr(x_np, dec_mlx_np, peak=2.0)
    pt_vs_input_psnr = _psnr(x_np, pt_dec_last, peak=2.0)

    print(
        f"[roundtrip] MLX-vs-input {mlx_vs_input_psnr:.2f} dB, "
        f"PT-vs-input {pt_vs_input_psnr:.2f} dB, "
        f"MLX-vs-PT {mlx_vs_pt_psnr:.2f} dB, "
        f"encode {t_enc:.2f}s decode {t_dec:.2f}s"
    )
    # Parity check: MLX must match the PyTorch reference within ~30 dB.
    assert mlx_vs_pt_psnr > 30.0, (
        f"MLX vs PT PSNR {mlx_vs_pt_psnr:.2f} dB < 30 dB (should be ≳50 dB)"
    )


def test_benchmark_5f_384(mlx_vae):
    """Peak-RAM + wall-time baseline for a 5-frame 384x384 clip."""
    import mlx.core as mx
    import resource

    T, H, W = 5, 384, 384
    x = mx.random.normal((1, 3, T, H, W)).astype(mx.bfloat16)
    mx.eval(x)

    peak0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    t0 = time.time()
    z = mlx_vae.encode(x)
    mx.eval(z)
    t_enc = time.time() - t0

    t0 = time.time()
    dec = mlx_vae.decode(z)
    mx.eval(dec)
    t_dec = time.time() - t0

    peak1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS returns bytes, Linux returns KB
    if sys.platform == "darwin":
        peak_gb = peak1 / (1024 ** 3)
    else:
        peak_gb = peak1 / (1024 ** 2)
    print(f"[bench 5f×{H}×{W}] encode {t_enc:.2f}s decode {t_dec:.2f}s "
          f"peak_rss={peak_gb:.2f} GB (delta {(peak1-peak0)/1e9:.2f} GB)")
    # No hard assertion -- purely informational for the port log.


if __name__ == "__main__":
    # Standalone runner (skips pytest fixture rigmarole for quick manual runs).
    _skip_if_missing()
    import mlx.core as mx
    from mlx_video.models.minimax_h3.video_vae import MiniMaxH3VideoVAE
    import torch
    sys.path.insert(0, str(PT_ROOT.parent))
    from video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE as PTVAE

    print("Loading MLX VAE...")
    m_mlx = MiniMaxH3VideoVAE()
    ck = mx.load(str(MLX_CKPT))
    m_mlx.load_weights(list(ck.items()), strict=False)

    print("Loading PyTorch VAE...")
    m_pt = PTVAE.from_pretrained(str(PT_ROOT)).eval().to(torch.float32)

    class Dummy:
        pass

    ctx_mlx = Dummy(); ctx_mlx.__class__ = type(m_mlx)
    ctx_pt = Dummy(); ctx_pt.__class__ = type(m_pt)

    test_encoder_parity(m_pt, m_mlx)
    test_decoder_parity(m_pt, m_mlx)
    test_full_round_trip_on_face(m_pt, m_mlx, tmp_path=Path("/tmp"))
    test_benchmark_5f_384(m_mlx)
