"""Phase 8-2: MLX Q4 quantize the H3 DiT.

Heavy Linear layers only:
  * blocks.*.attn.qkv_proj, .out_proj  (H3Attention)
  * blocks.*.mlp.fc1, .fc2             (H3MLP)
  * token_refiner.blocks.*.attn.qkv_proj, .out_proj
  * token_refiner.blocks.*.mlp.fc1, .fc2
  * condition_proj (5120 -> hidden), video_patch_proj, audio_patch_proj

Skips:
  * H3RMSNorm scales, all AdalnProj linears, TimeEmbedder linears,
    FinalLayer projections (small heads).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_video.models.minimax_h3.config import MiniMaxH3Config
from mlx_video.models.minimax_h3.model import MiniMaxH3Model


HEAVY_SUFFIX = (
    ".attn.qkv_proj", ".attn.out_proj",
    ".mlp.fc1", ".mlp.fc2",
    "condition_proj", "video_patch_proj", "audio_patch_proj",
)


def _predicate(path: str, module) -> bool:
    if not hasattr(module, "to_quantized"):
        return False
    if not any(path.endswith(s) for s in HEAVY_SUFFIX):
        return False
    # Skip Linears whose input dim doesn't divide group_size (small conditioning
    # projections like audio_patch_proj [32->5376] and video_patch_proj [96->5376]).
    w = getattr(module, "weight", None)
    if w is not None and w.ndim >= 2:
        in_dim = w.shape[-1]
        if in_dim % 64 != 0:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/mlx-video/mlx-models/MiniMaxH3-Ref2VA-MLX-bf16")
    ap.add_argument("--dst", default="~/mlx-video/mlx-models/MiniMaxH3-Ref2VA-MLX-Q4")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()

    dit_src = src / "dit" / "model.safetensors"
    dit_dst_dir = dst / "dit"
    dit_dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"[quantize] loading bf16 DiT from {dit_src} ...")
    t0 = time.time()
    cfg = MiniMaxH3Config()
    model = MiniMaxH3Model(cfg)
    model.load_weights(str(dit_src))
    print(f"[quantize] loaded in {time.time()-t0:.1f}s")

    # Count params before
    bf16_params = sum(p.size for _, p in tree_flatten(model.parameters()) if isinstance(p, mx.array))
    print(f"[quantize] pre-quant params: {bf16_params/1e9:.2f}B")

    print(f"[quantize] quantizing to {args.bits}-bit, group_size={args.group_size} ...")
    t0 = time.time()
    nn.quantize(model, group_size=args.group_size, bits=args.bits,
                class_predicate=_predicate)
    mx.eval(model.parameters())
    print(f"[quantize] quantized in {time.time()-t0:.1f}s")

    # Count params after quant (weight is now packed uint32; scales + biases added)
    def _bytes_of(x):
        if not isinstance(x, mx.array):
            return 0
        return x.size * x.dtype.size
    total_bytes = sum(_bytes_of(p) for _, p in tree_flatten(model.parameters()))
    print(f"[quantize] post-quant total tensor bytes: {total_bytes/1e9:.2f} GB")

    # Save
    out_file = dit_dst_dir / "model.safetensors"
    print(f"[quantize] saving to {out_file} ...")
    t0 = time.time()
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(out_file), flat)
    print(f"[quantize] saved in {time.time()-t0:.1f}s")

    # Copy the VAE/audio-VAE symlinks (or point to them)
    for sub in ("video_vae", "audio_vae"):
        src_link = src / sub
        dst_link = dst / sub
        if not dst_link.exists():
            dst_link.symlink_to(src_link.resolve())
            print(f"[quantize] symlinked {sub} -> {src_link.resolve()}")

    # Save quantization metadata alongside
    meta_file = dit_dst_dir / "quantization.json"
    import json
    meta_file.write_text(json.dumps({
        "bits": args.bits,
        "group_size": args.group_size,
        "class_predicate_suffixes": list(HEAVY_SUFFIX),
        "mode": "affine",
    }, indent=2))
    print(f"[quantize] wrote {meta_file}")

    on_disk = out_file.stat().st_size
    print(f"[quantize] on-disk safetensors: {on_disk/1e9:.2f} GB")


if __name__ == "__main__":
    main()
