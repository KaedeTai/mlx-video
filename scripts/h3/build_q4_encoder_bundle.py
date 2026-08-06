"""Phase 8.9-c: Build a persistent Q4 H3 text-encoder bundle."""
from __future__ import annotations

import argparse
import json
import shutil
import resource
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten as _tree_flatten


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3


TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "added_tokens.json",
    "merges.txt",
    "generation_config.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/mlx-video/mlx-models/H3-TextEncoder-MLX-bf16")
    ap.add_argument("--dst", default="~/mlx-video/mlx-models/H3-TextEncoder-MLX-Q4")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--shard-gb", type=float, default=4.5)
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[build-q4] src={src}")
    print(f"[build-q4] dst={dst}")
    print(f"[build-q4] bits={args.bits} group_size={args.group_size}")
    print(f"[build-q4] start rss={_rss_gb():.2f} GB", flush=True)

    from mlx_video.models.minimax_h3.text_encoder_bridge import H3TextEncoderBridge

    t0 = time.time()
    enc = H3TextEncoderBridge(model_path=str(src), load_vision=False)
    print(f"[build-q4] bf16 encoder loaded in {time.time()-t0:.1f}s, "
          f"rss={_rss_gb():.2f} GB", flush=True)

    t0 = time.time()
    nn.quantize(enc._lang, group_size=args.group_size, bits=args.bits, mode="affine")
    mx.eval(enc._lang.parameters())
    print(f"[build-q4] Q4 quantized in {time.time()-t0:.1f}s, "
          f"rss={_rss_gb():.2f} GB", flush=True)

    t0 = time.time()
    test_ctx = enc.encode("測試 quantization。", has_ref_image=True, has_ref_audio=True)
    mx.eval(test_ctx)
    print(f"[build-q4] sanity encode ok, shape={test_ctx.shape} "
          f"norm={float(mx.linalg.norm(test_ctx)):.3f} in {time.time()-t0:.2f}s",
          flush=True)

    weights = dict(_tree_flatten(enc._model.parameters()))
    n_keys = len(weights)
    total_bytes = sum(w.nbytes for w in weights.values())
    print(f"[build-q4] extracted {n_keys} tensors, "
          f"{total_bytes/1e9:.2f} GB", flush=True)

    sample_keys = sorted(weights.keys())[:5] + sorted(weights.keys())[-3:]
    for k in sample_keys:
        print(f"[build-q4]   sample key: {k}  shape={weights[k].shape} "
              f"dtype={weights[k].dtype}")

    shard_max_bytes = int(args.shard_gb * (1024**3))
    shard_idx = 1
    shard_state: dict[str, mx.array] = {}
    shard_bytes = 0
    weight_map: dict[str, str] = {}

    def _flush():
        nonlocal shard_idx, shard_state, shard_bytes
        if not shard_state:
            return
        tmp_name = f"model-part-{shard_idx:03d}.safetensors"
        tmp_path = dst / tmp_name
        print(f"[build-q4]   flush shard {shard_idx}: {len(shard_state)} tensors, "
              f"{shard_bytes/1e9:.2f} GB -> {tmp_name}", flush=True)
        mx.save_safetensors(str(tmp_path), shard_state)
        shard_state = {}
        shard_bytes = 0
        shard_idx += 1

    ordered_keys = sorted(weights.keys())
    for k in ordered_keys:
        arr = weights[k]
        b = int(arr.nbytes)
        if shard_bytes > 0 and shard_bytes + b > shard_max_bytes:
            _flush()
        shard_state[k] = arr
        shard_bytes += b
        weight_map[k] = f"model-part-{shard_idx:03d}.safetensors"
    _flush()

    total_shards = shard_idx - 1
    print(f"[build-q4] {total_shards} shards; renaming...")
    rename_map = {}
    for i in range(1, total_shards + 1):
        old = f"model-part-{i:03d}.safetensors"
        new = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        (dst / old).rename(dst / new)
        rename_map[old] = new
    weight_map = {k: rename_map[v] for k, v in weight_map.items()}

    cfg = json.loads((src / "config.json").read_text())
    quant_block = {"group_size": args.group_size, "bits": args.bits}
    cfg["quantization"] = quant_block
    tc = cfg.setdefault("text_config", {})
    tc["quantization"] = quant_block
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[build-q4] wrote config.json with quantization={quant_block}")

    for name in TOKENIZER_FILES:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)
            print(f"[build-q4] copied {name}")

    index = {
        "metadata": {
            "total_size": total_bytes,
            "source_bf16": str(src),
            "phase": "8.9-c",
            "quantization": quant_block,
            "quantization_mode": "affine",
            "note": "H3-tuned encoder, Q4 quantized language stack. "
                    "Vision tower + lm_head + final norm dropped.",
        },
        "weight_map": weight_map,
    }
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"[build-q4] wrote index.json ({len(weight_map)} keys)")

    on_disk = sum(f.stat().st_size for f in dst.rglob('*') if f.is_file())
    print(f"[build-q4] DONE dst_size={on_disk/1e9:.2f} GB, "
          f"peak_rss={_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
