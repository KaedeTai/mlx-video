"""Phase 8.9-b: Convert Comfy-Org/MiniMax-H3 text-encoder checkpoint into an
mlx-vlm-compatible model directory.

Source
------
Single file: ~/models/H3-text-encoder-bf16/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors
- 51.5 GB, 902 tensors, bf16
- Comfy naming:  model.embed_tokens.*, model.layers.[0..49].*, visual.*
- __metadata__: {'minimax_h3_te': '{"num_hidden_layers": 50,
                                      "output": "unnormalized_hidden_after_layer_50"}'}

Target
------
Directory at --out (default ~/mlx-video/mlx-models/H3-TextEncoder-MLX-bf16/):

  config.json                  <-- copied from Q4 template, patched:
                                     text_config.num_hidden_layers = 50
                                     text_config.dtype             = bfloat16
                                     quantization / quantization_config removed
  tokenizer.json               <-- copied from Q4 template
  tokenizer_config.json
  preprocessor_config.json
  chat_template.jinja
  chat_template.json
  added_tokens.json
  merges.txt
  model-000XX-of-000NN.safetensors ...  <-- rekeyed + sharded
  model.safetensors.index.json

Key rekey (matches mlx_vlm.models.qwen3_vl.Model.sanitize expectations):

  Comfy   `model.embed_tokens.*`           ->  `model.language_model.embed_tokens.*`
  Comfy   `model.layers.N.*`               ->  `model.language_model.layers.N.*`
  Comfy   `visual.*`                       ->  `model.visual.*`         (dropped if --no-vision)

After mlx_vlm.sanitize:
  model.language_model.*  ->  language_model.model.*   (final MLX key)
  model.visual.*          ->  vision_tower.*

Truncation
----------
The checkpoint is already truncated to 50 layers by MiniMax. We only need to
patch config.text_config.num_hidden_layers to 50 so mlx_vlm allocates the
right number of decoder blocks.

Usage:
  python -m scripts.h3.convert_h3_text_encoder \
      --src ~/models/H3-text-encoder-bf16/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors \
      --template ~/models/Qwen3-VL-32B-Instruct-4bit \
      --out ~/mlx-video/mlx-models/H3-TextEncoder-MLX-bf16 \
      --shard-gb 5 [--no-vision]
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save
import torch


# ---------------------------------------------------------------------------
# Key rekey (Comfy -> pre-sanitize mlx-vlm)
# ---------------------------------------------------------------------------

def _rekey(comfy_key: str, keep_vision: bool) -> str | None:
    """Return the mlx-vlm pre-sanitize key, or None to drop the tensor."""
    if comfy_key.startswith("model."):
        # model.embed_tokens.* / model.layers.N.* / model.norm.weight -> language_model
        return "model.language_model." + comfy_key[len("model."):]
    if comfy_key.startswith("visual."):
        if not keep_vision:
            return None
        return "model.visual." + comfy_key[len("visual."):]
    # unknown prefix — pass through
    return comfy_key


# ---------------------------------------------------------------------------
# Config patch
# ---------------------------------------------------------------------------

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


def _patch_config(template_cfg: dict, keep_vision: bool) -> dict:
    cfg = json.loads(json.dumps(template_cfg))  # deep copy

    # top-level quantization keys — drop
    cfg.pop("quantization", None)
    cfg.pop("quantization_config", None)

    # text_config: truncate to 50 layers, set dtype bf16
    tc = cfg.setdefault("text_config", {})
    tc["num_hidden_layers"] = 50
    tc["dtype"] = "bfloat16"
    tc.pop("quantization", None)
    tc.pop("quantization_config", None)

    # vision_config: keep unchanged (Qwen3-VL vision tower is standalone)
    vc = cfg.get("vision_config", {})
    if vc:
        vc.pop("quantization", None)
        vc.pop("quantization_config", None)

    if not keep_vision:
        # Leave vision_config in place so mlx-vlm's Model.__init__ won't crash,
        # but mark: caller is expected to del model.vision_tower after load,
        # OR set vision_config to a minimal skeleton. mlx-vlm's Model tries
        # to instantiate VisionModel unconditionally, so best is to leave
        # vision_config intact and drop the tower weights externally.
        pass

    cfg["dtype"] = "bfloat16"
    return cfg


# ---------------------------------------------------------------------------
# Streaming shard writer
# ---------------------------------------------------------------------------

def _bytes_of(arr: torch.Tensor) -> int:
    return arr.numel() * arr.element_size()


def convert(src: Path, template: Path, out: Path,
            shard_gb: float = 5.0, keep_vision: bool = True) -> dict:
    src = Path(src).expanduser().resolve()
    template = Path(template).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[convert] src={src}")
    print(f"[convert] template={template}")
    print(f"[convert] out={out}")
    print(f"[convert] keep_vision={keep_vision}, shard_gb={shard_gb}")

    # ---- 1) copy tokenizer + auxiliary files ----
    for name in TOKENIZER_FILES:
        p = template / name
        if p.exists():
            shutil.copy2(p, out / name)
            print(f"[convert] copied {name}")

    # ---- 2) patch and write config.json ----
    tpl_cfg = json.loads((template / "config.json").read_text())
    new_cfg = _patch_config(tpl_cfg, keep_vision=keep_vision)
    (out / "config.json").write_text(json.dumps(new_cfg, indent=2))
    print(f"[convert] wrote config.json (num_hidden_layers=50, no-quant)")

    # ---- 3) stream tensors: read from src, rekey, shard, write ----
    shard_max_bytes = int(shard_gb * (1024 ** 3))

    shard_idx = 1
    shard_bytes = 0
    shard_state: dict[str, torch.Tensor] = {}
    weight_map: dict[str, str] = {}
    total_bytes = 0
    num_tensors = 0
    num_dropped = 0

    def _flush(final: bool = False):
        nonlocal shard_idx, shard_state, shard_bytes
        if not shard_state:
            return
        # Filename: we don't yet know NN (total shards); use a two-pass rename
        # at the end. For now use a placeholder.
        tmp_name = f"model-part-{shard_idx:03d}.safetensors"
        tmp_path = out / tmp_name
        print(f"[convert]   flush shard {shard_idx}: {len(shard_state)} tensors, "
              f"{shard_bytes / 1e9:.2f} GB -> {tmp_name}")
        safetensors_save(shard_state, str(tmp_path))
        shard_state = {}
        shard_bytes = 0
        shard_idx += 1

    with safe_open(str(src), framework="pt", device="cpu") as f:
        src_keys = list(f.keys())
        print(f"[convert] src has {len(src_keys)} tensors")

        for k in src_keys:
            new_k = _rekey(k, keep_vision=keep_vision)
            if new_k is None:
                num_dropped += 1
                continue
            t = f.get_tensor(k)  # keeps original dtype (bf16 -> torch.bfloat16)
            b = _bytes_of(t)
            if shard_bytes > 0 and shard_bytes + b > shard_max_bytes:
                _flush()
            # Remember which shard this key is going into (placeholder name)
            weight_map[new_k] = f"model-part-{shard_idx:03d}.safetensors"
            shard_state[new_k] = t
            shard_bytes += b
            total_bytes += b
            num_tensors += 1
            if num_tensors % 100 == 0:
                print(f"[convert]   progress: {num_tensors} tensors "
                      f"({total_bytes / 1e9:.2f} GB), shard {shard_idx} "
                      f"({shard_bytes / 1e9:.2f} GB in progress)")
        _flush(final=True)

    total_shards = shard_idx - 1
    # ---- 4) rename placeholder shards to model-000XX-of-000NN.safetensors ----
    print(f"[convert] {total_shards} shards; renaming...")
    rename_map = {}
    for i in range(1, total_shards + 1):
        old = f"model-part-{i:03d}.safetensors"
        new = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        (out / old).rename(out / new)
        rename_map[old] = new
    # patch weight_map
    weight_map = {k: rename_map[v] for k, v in weight_map.items()}

    # ---- 5) write model.safetensors.index.json ----
    index = {
        "metadata": {
            "total_size": total_bytes,
            "source": str(src),
            "phase": "8.9-b",
            "minimax_h3_te": '{"num_hidden_layers": 50, '
                             '"output": "unnormalized_hidden_after_layer_50"}',
            "note": "Comfy-Org bf16 rekeyed for mlx-vlm. Vision tower "
                    "included=" + str(keep_vision),
        },
        "weight_map": weight_map,
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"[convert] wrote model.safetensors.index.json "
          f"({len(weight_map)} keys)")

    stats = {
        "wall_time_s": time.time() - t0,
        "total_bytes": total_bytes,
        "total_gb": total_bytes / 1e9,
        "num_tensors": num_tensors,
        "num_dropped": num_dropped,
        "num_shards": total_shards,
        "keep_vision": keep_vision,
    }
    print(f"[convert] DONE {stats}")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src",
                   default="~/models/H3-text-encoder-bf16/text_encoders/"
                           "qwen3vl_32b_minimax_h3_bf16.safetensors")
    p.add_argument("--template", default="~/models/Qwen3-VL-32B-Instruct-4bit",
                   help="dir to copy tokenizer + config.json template from")
    p.add_argument("--out", default="~/mlx-video/mlx-models/H3-TextEncoder-MLX-bf16")
    p.add_argument("--shard-gb", type=float, default=5.0)
    p.add_argument("--no-vision", action="store_true",
                   help="drop visual.* tensors (saves ~4 GB, but mlx-vlm still "
                        "instantiates the vision tower; delete after load)")
    args = p.parse_args()

    convert(Path(args.src), Path(args.template), Path(args.out),
            shard_gb=args.shard_gb, keep_vision=not args.no_vision)


if __name__ == "__main__":
    main()
