"""Weight converter for the MiniMax H3 stack.

Phase 2: **Video VAE only**. Reads the PyTorch safetensors at

    ~/models/MiniMax-H3-raw/Ref2VA/video_vae/source/model.safetensors

and writes an MLX-native safetensors file whose keys map 1:1 onto the MLX
:class:`MiniMaxH3VideoVAE` module hierarchy (so ``model.load_weights(path)``
just works).

Layout transforms applied:

* ``Conv3d`` weights: PyTorch ``(O, I, D, H, W)`` -> MLX ``(O, D, H, W, I)``
* Everything else is copied verbatim.

Precision: source is fp32. We emit bf16 by default and can also produce fp16
for quicker iteration.

Later phases will grow this file with converters for transformer, audio VAE,
and the trimmed text encoder (see PORT_PLAN.md §4-6).

Run directly::

    python -m mlx_video.models.minimax_h3.convert --dtype bf16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import safetensors.torch


# Every conv3d weight key ends with one of these tails (bias tensors are 1-D
# and skipped naturally by the shape check inside :func:`_is_conv3d_weight`).
CONV3D_WEIGHT_TAILS = (
    "conv_in.weight",
    "conv_out.weight",
    "conv1.weight",
    "conv2.weight",
    "nin_shortcut.weight",
    "downsample.conv.weight",
    "quant_conv.weight",
    "post_quant_conv.weight",
)


def _is_conv3d_weight(key: str, arr: np.ndarray) -> bool:
    if arr.ndim != 5:
        return False
    return any(key.endswith(tail) for tail in CONV3D_WEIGHT_TAILS)


def _to_mlx_layout(key: str, arr: np.ndarray) -> np.ndarray:
    if _is_conv3d_weight(key, arr):
        return np.ascontiguousarray(arr.transpose(0, 2, 3, 4, 1))
    return arr


def convert_video_vae(
    src: Path,
    dst: Path,
    dtype: str = "bf16",
    verify_keys: bool = True,
) -> dict:
    """Convert one Ref2VA video-VAE safetensors file to MLX layout.

    Returns a stats dict for logging.
    """
    src = Path(src).expanduser()
    dst = Path(dst).expanduser()
    dst.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sd = safetensors.torch.load_file(str(src))

    out: dict[str, mx.array] = {}
    total_params = 0
    num_transposed = 0
    for k, v in sd.items():
        arr = v.detach().cpu().numpy()
        if _is_conv3d_weight(k, arr):
            num_transposed += 1
        arr = _to_mlx_layout(k, arr)

        a = mx.array(arr)
        if dtype == "bf16":
            a = a.astype(mx.bfloat16)
        elif dtype == "fp16":
            a = a.astype(mx.float16)
        elif dtype == "fp32":
            a = a.astype(mx.float32)
        else:
            raise ValueError(f"unsupported dtype: {dtype}")
        out[k] = a
        total_params += arr.size

    if verify_keys:
        try:
            from mlx.utils import tree_flatten
            from .video_vae import MiniMaxH3VideoVAE
            m = MiniMaxH3VideoVAE()
            mlx_keys = {k for k, _ in tree_flatten(m.parameters())}
            missing = set(out) - mlx_keys
            extra = mlx_keys - set(out)
            # These are computed inside __init__, not stored in ckpt.
            allowed_extra = {
                "latents_mean", "latents_std", "pixel_mean", "pixel_std",
                "decoder.pos_embed.inv_freq",
            }
            unexplained = extra - allowed_extra
            if missing or unexplained:
                print(f"[convert] WARN key mismatch: {len(missing)} ckpt-only, "
                      f"{len(unexplained)} module-only")
                for k in sorted(missing)[:10]:
                    print(f"  ckpt-only: {k}")
                for k in sorted(unexplained)[:10]:
                    print(f"  module-only: {k}")
        except Exception as e:
            print(f"[convert] key verify skipped: {e}")

    mx.save_safetensors(str(dst), out)
    dt = time.time() - t0
    size_mb = dst.stat().st_size / 1024 / 1024

    return dict(
        num_tensors=len(out),
        num_params=total_params,
        num_transposed=num_transposed,
        elapsed=dt,
        output_mb=size_mb,
    )

# ---------------------------------------------------------------------------
# Phase 3: Audio VAE converter (weight-norm folding + Conv1d layout swap)
# ---------------------------------------------------------------------------


def _fold_weight_norm(sd: dict) -> int:
    """In-place fold every `.weight_g` / `.weight_v` pair to a plain `.weight`.

    PyTorch parametrized weight-norm stores
        weight_g : (O, 1, 1)          per-out-channel scale
        weight_v : same shape as w    unnormalized direction
    and the effective weight is
        w = g * v / ||v||_2         with ||·||_2 taken across all dims except 0.
    Returns the number of pairs folded.
    """
    import numpy as np

    folded = 0
    keys = list(sd.keys())
    for k in keys:
        if not k.endswith(".weight_g"):
            continue
        base = k[: -len(".weight_g")]
        vk = f"{base}.weight_v"
        if vk not in sd:
            continue
        g = sd[k].detach().cpu().float().numpy()   # (O, 1, 1)
        v = sd[vk].detach().cpu().float().numpy()  # (O, I, K)
        # PyTorch weight_norm normalizes over dims [1..ndim-1] (all but the
        # 0-th output dim). Keep dim 0 so the broadcast with g works.
        axes = tuple(range(1, v.ndim))
        v_norm = np.sqrt((v * v).sum(axis=axes, keepdims=True))
        w = g * v / (v_norm + 1e-12)
        # Preserve source dtype
        w_t = safetensors.torch.torch.from_numpy(w.astype(np.float32)).to(sd[vk].dtype)
        sd[base + ".weight"] = w_t
        del sd[k]
        del sd[vk]
        folded += 1
    return folded


# Every 1D conv weight key. Bias is 1-D and skipped by the shape check inside
# _to_mlx_layout_1d.  ConvTranspose1d in the source has weight shape (I, O, K)
# which is *distinguishable* from Conv1d (O, I, K) only by key name — we list
# them explicitly.  Attention-projection weights (LinearWithoutBias.qkv, proj)
# are 2-D (O, I) and need no permutation.

# Keys whose PyTorch weight layout is (I, O, K) rather than (O, I, K).
CONVTRANSPOSE1D_KEY_SUFFIXES = tuple(f"decoder.ups.{i}.0.weight" for i in range(7))


def _is_conv1d_weight(key: str, arr) -> bool:
    return arr.ndim == 3 and key.endswith(".weight") and not key.endswith("filter")


def _is_convtranspose1d_weight(key: str, arr) -> bool:
    return arr.ndim == 3 and any(key.endswith(s) for s in CONVTRANSPOSE1D_KEY_SUFFIXES)


def _is_filter_buffer(key: str, arr) -> bool:
    return arr.ndim == 3 and key.endswith(".filter")


def _to_mlx_layout_1d(key: str, arr):
    """Permute source-layout weights to MLX layout for 1D convs.

    * PyTorch Conv1d          (O, I, K) → MLX (O, K, I) = perm (0, 2, 1)
    * PyTorch ConvTranspose1d (I, O, K) → MLX (O, K, I) = perm (1, 2, 0)
    * Filter buffers          (1, 1, K) → keep as-is (expanded at runtime)
    """
    import numpy as np

    if _is_convtranspose1d_weight(key, arr):
        return np.ascontiguousarray(arr.transpose(1, 2, 0))
    if _is_conv1d_weight(key, arr):
        return np.ascontiguousarray(arr.transpose(0, 2, 1))
    return arr


def convert_audio_vae(
    src: Path,
    dst: Path,
    dtype: str = "bf16",
    verify_keys: bool = True,
) -> dict:
    """Convert the Ref2VA audio-VAE safetensors to MLX layout with weight-norm folded."""
    import numpy as np

    src = Path(src).expanduser()
    dst = Path(dst).expanduser()
    dst.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sd = safetensors.torch.load_file(str(src))

    n_folded = _fold_weight_norm(sd)

    # latents_mean/std live in config.json for this checkpoint, not the
    # safetensors. Add them so the converted file is self-contained.
    from .audio_vae import LATENTS_MEAN, LATENTS_STD
    import torch as _torch
    sd["latents_mean"] = _torch.tensor(LATENTS_MEAN, dtype=_torch.float32)
    sd["latents_std"] = _torch.tensor(LATENTS_STD, dtype=_torch.float32)

    out: dict[str, mx.array] = {}
    total_params = 0
    n_conv1d = 0
    n_convt1d = 0
    for k, v in sd.items():
        arr = v.detach().cpu().float().numpy()
        if _is_convtranspose1d_weight(k, arr):
            n_convt1d += 1
        elif _is_conv1d_weight(k, arr):
            n_conv1d += 1
        arr = _to_mlx_layout_1d(k, arr)

        a = mx.array(arr)
        if dtype == "bf16":
            a = a.astype(mx.bfloat16)
        elif dtype == "fp16":
            a = a.astype(mx.float16)
        elif dtype == "fp32":
            a = a.astype(mx.float32)
        else:
            raise ValueError(f"unsupported dtype: {dtype}")
        out[k] = a
        total_params += arr.size

    if verify_keys:
        try:
            from mlx.utils import tree_flatten
            from .audio_vae import MiniMaxH3AudioVAE
            m = MiniMaxH3AudioVAE()
            mlx_keys = {k for k, _ in tree_flatten(m.parameters())}
            missing = set(out) - mlx_keys
            extra = mlx_keys - set(out)
            # latents_mean/std come from config.json, not the safetensors.
            allowed_extra = {"latents_mean", "latents_std"}
            unexplained = extra - allowed_extra
            if missing or unexplained:
                print(f"[convert-audio] WARN key mismatch: {len(missing)} ckpt-only, "
                      f"{len(unexplained)} module-only")
                for k in sorted(missing)[:15]:
                    print(f"  ckpt-only: {k}  shape={tuple(out[k].shape)}")
                for k in sorted(unexplained)[:15]:
                    print(f"  module-only: {k}")
        except Exception as e:
            print(f"[convert-audio] key verify skipped: {e}")

    mx.save_safetensors(str(dst), out)
    dt = time.time() - t0
    size_mb = dst.stat().st_size / 1024 / 1024

    return dict(
        num_tensors=len(out),
        num_params=total_params,
        num_weight_norm_folded=n_folded,
        num_conv1d_permuted=n_conv1d,
        num_convtranspose1d_permuted=n_convt1d,
        elapsed=dt,
        output_mb=size_mb,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--component", choices=("video-vae", "audio-vae"), default="video-vae")
    p.add_argument("--src", default=None)
    p.add_argument("--dst", default=None)
    p.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    args = p.parse_args()

    if args.component == "video-vae":
        src = args.src or "~/models/MiniMax-H3-raw/Ref2VA/video_vae/source/model.safetensors"
        dst = args.dst or "~/mlx-video/mlx-models/MiniMaxH3-VideoVAE-MLX-bf16/model.safetensors"
        stats = convert_video_vae(Path(src), Path(dst), dtype=args.dtype)
        print(
            f"[convert] {stats['num_tensors']} tensors ({stats['num_params']:,} params), "
            f"{stats['num_transposed']} conv3d transposed, "
            f"{stats['output_mb']:.1f} MB in {stats['elapsed']:.1f}s -> {dst}"
        )
    else:
        src = args.src or "~/models/MiniMax-H3-raw/Ref2VA/audio_vae/model.safetensors"
        dst = args.dst or "~/mlx-video/mlx-models/MiniMaxH3-AudioVAE-MLX-bf16/model.safetensors"
        stats = convert_audio_vae(Path(src), Path(dst), dtype=args.dtype)
        print(
            f"[convert-audio] {stats['num_tensors']} tensors ({stats['num_params']:,} params), "
            f"WN folded={stats['num_weight_norm_folded']}, "
            f"conv1d permuted={stats['num_conv1d_permuted']}, "
            f"convT1d permuted={stats['num_convtranspose1d_permuted']}, "
            f"{stats['output_mb']:.1f} MB in {stats['elapsed']:.1f}s -> {dst}"
        )


if __name__ == "__main__":
    main()


