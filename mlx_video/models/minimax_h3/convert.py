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


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        default="~/models/MiniMax-H3-raw/Ref2VA/video_vae/source/model.safetensors",
    )
    p.add_argument(
        "--dst",
        default="~/mlx-video/mlx-models/MiniMaxH3-VideoVAE-MLX-bf16/model.safetensors",
    )
    p.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    args = p.parse_args()

    stats = convert_video_vae(Path(args.src), Path(args.dst), dtype=args.dtype)
    print(
        f"[convert] {stats['num_tensors']} tensors ({stats['num_params']:,} params), "
        f"{stats['num_transposed']} conv3d transposed, "
        f"{stats['output_mb']:.1f} MB in {stats['elapsed']:.1f}s -> {args.dst}"
    )


if __name__ == "__main__":
    main()
