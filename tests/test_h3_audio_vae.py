"""Numerical parity: MLX H3 Audio VAE vs PyTorch reference (Ref2VA/audio_vae).

Loads the bundled ``DacAudioVAE`` (which the checkpoint's ``auto_map`` normally
resolves via ``trust_remote_code`` in diffusers) directly out of the raw model
folder, then reconstructs the ComfyUI-style ``encode`` and ``decode`` functions
around it — the reference bundle only exposes ``decode``.

The test is skipped if the raw PyTorch checkpoint or the MLX-converted
safetensors are missing.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest


MLX_CKPT = Path("~/mlx-video/mlx-models/MiniMaxH3-AudioVAE-MLX-bf16/model.safetensors").expanduser()
PT_ROOT = Path("~/models/MiniMax-H3-raw/Ref2VA/audio_vae").expanduser()


def _skip_if_missing():
    if not MLX_CKPT.is_file():
        pytest.skip(f"missing MLX checkpoint: {MLX_CKPT}")
    if not (PT_ROOT / "model.safetensors").is_file():
        pytest.skip(f"missing PyTorch VAE at {PT_ROOT}")


def _psnr(a: np.ndarray, b: np.ndarray, peak: float | None = None) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if peak is None:
        peak = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-9) * 2.0
    if mse == 0:
        return math.inf
    return 20.0 * math.log10(peak / math.sqrt(mse))


# ---------------------------------------------------------------------------
# Load the bundled DacAudioVAE (relative imports rewritten to package form)
# ---------------------------------------------------------------------------


def _load_pt_bundle():
    """Import DacAudioVAE from the raw checkpoint folder as ``h3_ref.*``."""
    pkg = types.ModuleType("h3_ref")
    pkg.__path__ = [str(PT_ROOT)]
    sys.modules["h3_ref"] = pkg

    shim_dir = Path("/tmp/h3_ref_shim")
    shim_dir.mkdir(exist_ok=True)
    for mn in [
        "dac_utils", "dac_activations", "dac_alias_free_filter",
        "dac_alias_free_resample", "dac_alias_free_act",
        "dac_attn_proj", "dac_bigvgan", "dac_audio_vae",
    ]:
        src = (PT_ROOT / f"{mn}.py").read_text().replace("from .", "from h3_ref.")
        shim = shim_dir / f"{mn}.py"
        shim.write_text(src)
        spec = importlib.util.spec_from_file_location(f"h3_ref.{mn}", shim)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"h3_ref.{mn}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["h3_ref.dac_audio_vae"].DacAudioVAE


# ---------------------------------------------------------------------------
# ComfyUI-style encode / decode wrappers around the bundled `DacAudioVAE`
# ---------------------------------------------------------------------------

from mlx_video.models.minimax_h3.audio_vae import LATENTS_MEAN, LATENTS_STD


def _pt_encode(model, waveform, mean_t, std_t):
    import torch
    b, s, length = waveform.shape
    hop = model.hop_length
    right_pad = math.ceil(length / hop) * hop - length
    if right_pad > 0:
        waveform = torch.nn.functional.pad(waveform, (0, right_pad))
    x = waveform.reshape(b * s, 1, -1)
    x = model.encoder(x)
    x = model.pre_block(x.transpose(1, 2)).transpose(1, 2)
    z = model.mean_proj(x)
    z = (z - mean_t) / std_t
    return z.reshape(b, s, z.shape[1], z.shape[2]).permute(0, 2, 1, 3)


def _pt_decode(model, z, mean_t, std_t):
    b, c, s, t = z.shape
    z = z.permute(0, 2, 1, 3).reshape(b * s, c, t)
    z = z * std_t + mean_t
    x = model.dec_in_proj(z)
    x = model.decoder(x)
    return x.reshape(b, s, -1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pt_vae():
    _skip_if_missing()
    import torch
    from safetensors.torch import load_file
    DacAudioVAE = _load_pt_bundle()
    model = DacAudioVAE(
        encoder_dim=64, encoder_rates=[2, 4, 4, 5, 5],
        decoder_dim=1024, decoder_rates=[5, 5, 2, 2, 2, 2, 2],
        attn_proj=True, decoder_type="bigvgan",
        vae_latent_channels=32, sample_rate=32000,
    )
    sd = load_file(str(PT_ROOT / "model.safetensors"))
    model.load_state_dict(sd, strict=True)
    model.eval()
    mean_t = torch.tensor(LATENTS_MEAN, dtype=torch.float32).view(1, -1, 1)
    std_t = torch.tensor(LATENTS_STD, dtype=torch.float32).view(1, -1, 1)
    return model, mean_t, std_t


@pytest.fixture(scope="module")
def mlx_vae():
    _skip_if_missing()
    import mlx.core as mx
    from mlx_video.models.minimax_h3.audio_vae import MiniMaxH3AudioVAE
    m = MiniMaxH3AudioVAE()
    ck = mx.load(str(MLX_CKPT))
    # Cast bf16 storage to fp32 for the parity comparison (matches PT compute
    # precision so we only measure architectural equivalence).
    ck_fp32 = {k: v.astype(mx.float32) for k, v in ck.items()}
    m.load_weights(list(ck_fp32.items()))
    return m


def _make_signal(sr=32000, seconds=1.0, seed=0):
    """Test signal: sine sweep + noise, stereo, in [-1, 1]."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    # 200 Hz → 8 kHz log sweep
    phase = 2 * np.pi * (200 * np.exp(np.log(8000 / 200) * t) * t)
    tone = 0.35 * np.sin(phase)
    noise = 0.03 * rng.standard_normal(n)
    left = tone + noise
    right = np.roll(tone, sr // 200) * 0.9 + 0.03 * rng.standard_normal(n)
    return np.stack([left, right])[None, ...].astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_latent_shape_and_range(mlx_vae):
    import mlx.core as mx
    signal = _make_signal(seconds=3.0)  # 3 s → 40 fps × 3 = 120 latent frames
    x = mx.array(signal)
    z = mlx_vae.encode(x)
    mx.eval(z)
    # (B=1, C=32, S=2, T=120)
    assert z.shape == (1, 32, 2, 120), f"unexpected latent shape {z.shape}"
    z_np = np.array(z)
    assert -6 < z_np.min() < z_np.max() < 6, f"latents look off: [{z_np.min()}, {z_np.max()}]"


def test_encode_parity(mlx_vae, pt_vae):
    import mlx.core as mx
    import torch
    pt_model, mean_t, std_t = pt_vae
    signal = _make_signal(seconds=1.0)
    with torch.inference_mode():
        z_pt = _pt_encode(pt_model, torch.from_numpy(signal), mean_t, std_t).numpy()
    z_mlx = np.array(mlx_vae.encode(mx.array(signal)))
    assert z_pt.shape == z_mlx.shape
    dB = _psnr(z_pt, z_mlx)
    print(f"\nencode PSNR: {dB:.2f} dB  (max abs diff {np.abs(z_pt - z_mlx).max():.2e})")
    assert dB > 30.0, f"encode parity below 30 dB: {dB:.2f}"


def test_decoder_parity(mlx_vae, pt_vae):
    """Decoder-only parity: feed the PT-computed z to both decoders."""
    import mlx.core as mx
    import torch
    pt_model, mean_t, std_t = pt_vae
    signal = _make_signal(seconds=1.0)
    with torch.inference_mode():
        z_pt = _pt_encode(pt_model, torch.from_numpy(signal), mean_t, std_t)
        y_pt = _pt_decode(pt_model, z_pt, mean_t, std_t).numpy()
    y_mlx = np.array(mlx_vae.decode(mx.array(z_pt.numpy())))
    dB = _psnr(y_pt, y_mlx, peak=2.0)
    print(f"\ndecoder-only PSNR: {dB:.2f} dB (max abs diff {np.abs(y_pt - y_mlx).max():.2e})")
    assert dB > 30.0, f"decoder parity below 30 dB: {dB:.2f}"


def test_end_to_end_parity(mlx_vae, pt_vae):
    """MLX (encode+decode) vs PyTorch (encode+decode) on the same input."""
    import mlx.core as mx
    import torch
    pt_model, mean_t, std_t = pt_vae
    signal = _make_signal(seconds=1.0)
    with torch.inference_mode():
        z_pt = _pt_encode(pt_model, torch.from_numpy(signal), mean_t, std_t)
        y_pt = _pt_decode(pt_model, z_pt, mean_t, std_t).numpy()
    x_mlx = mx.array(signal)
    y_mlx = np.array(mlx_vae.decode(mlx_vae.encode(x_mlx)))
    dB = _psnr(y_pt, y_mlx, peak=2.0)
    print(f"\nend-to-end PSNR (peak=2): {dB:.2f} dB  (max abs diff {np.abs(y_pt - y_mlx).max():.2e})")
    assert dB > 30.0, f"end-to-end parity below 30 dB: {dB:.2f}"


def test_output_in_range(mlx_vae):
    """decode() should always clamp to [-1, 1]."""
    import mlx.core as mx
    signal = _make_signal(seconds=1.0)
    y = np.array(mlx_vae.decode(mlx_vae.encode(mx.array(signal))))
    assert y.min() >= -1.0 - 1e-5 and y.max() <= 1.0 + 1e-5, \
        f"decoder output out of range: [{y.min()}, {y.max()}]"
