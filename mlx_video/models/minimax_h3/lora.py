"""H3 Turbo LoRA runtime overlay loader.

Loads LoRA A/B pairs from a safetensors file and wraps target Linear /
QuantizedLinear modules in the DiT so that each forward call adds the LoRA
delta

    y += alpha * (x @ A.T) @ B.T

without modifying the underlying (possibly Q4) base weights.  This is the
runtime-overlay strategy from the porting plan section 4 (Option B), chosen
here because it keeps the base Q4 weights intact and avoids a re-quantization
pass.

LoRA modules map 1:1 into ``MiniMaxH3Model``:

    blocks.N.attn.qkv_proj
    blocks.N.attn.out_proj
    blocks.N.mlp.fc1
    blocks.N.mlp.fc2
    blocks.N.adaln_proj.linear                        (N = 0..49)
    final_layer.adaln_proj.linear
    token_refiner.blocks.{0,1}.attn.{qkv_proj,out_proj}
    token_refiner.blocks.{0,1}.mlp.{fc1,fc2}

Total: 259 modules (259 A + 259 B tensors = 518 keys).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn


class _LoRAOverlay(nn.Module):
    """Wrap a base Linear / QuantizedLinear with a runtime LoRA branch.

    The base module's ``__call__`` is used unchanged; the LoRA delta is added
    in the ambient dtype of ``x`` (typically bfloat16 or float32).
    """

    def __init__(self, base: nn.Module, A: mx.array, B: mx.array, alpha: float):
        super().__init__()
        self.base = base
        # A: [r, in_features], B: [out_features, r] - kept in bf16.
        self.lora_A = A.astype(mx.bfloat16)
        self.lora_B = B.astype(mx.bfloat16)
        self.lora_alpha = float(alpha)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        A = self.lora_A.astype(x.dtype)
        B = self.lora_B.astype(x.dtype)
        # Two small matmuls: (x @ A.T) -> [..., r], then @ B.T -> [..., out].
        delta = (x @ A.T) @ B.T
        if self.lora_alpha != 1.0:
            delta = delta * self.lora_alpha
        return y + delta


def _resolve(root: nn.Module, parts):
    """Walk parts[:-1] through the module tree; return (parent, leaf_name)."""
    obj: Any = root
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj, parts[-1]


def load_turbo_lora(
    dit_model: nn.Module,
    lora_path,
    alpha: float = 1.0,
    verbose: bool = True,
) -> int:
    """Load a Turbo LoRA safetensors and install runtime overlays.

    Returns
    -------
    int : number of modules wrapped.
    """
    lora_path = Path(lora_path).expanduser()
    t0 = time.time()

    tensors = mx.load(str(lora_path))
    keys = list(tensors.keys())
    modules = sorted({k.rsplit(".lora_", 1)[0] for k in keys})

    if verbose:
        print(f"[H3-LoRA] loading {lora_path.name}: "
              f"{len(keys)} tensors -> {len(modules)} modules")

    counts = {"attn": 0, "mlp": 0, "adaln": 0, "refiner": 0, "final": 0, "other": 0}
    wrapped = 0
    for mod_path in modules:
        a_key = mod_path + ".lora_A.weight"
        b_key = mod_path + ".lora_B.weight"
        if a_key not in tensors or b_key not in tensors:
            raise KeyError(f"missing LoRA pair for {mod_path}")
        A = tensors[a_key]
        B = tensors[b_key]

        parts = mod_path.split(".")
        try:
            parent, leaf = _resolve(dit_model, parts)
            base = getattr(parent, leaf)
        except (AttributeError, IndexError, KeyError) as e:
            raise RuntimeError(
                f"[H3-LoRA] could not resolve module '{mod_path}' in DiT: {e}"
            ) from e

        # Shape sanity check against the base weight (handle Q4 packed layout).
        base_w = getattr(base, "weight", None)
        if base_w is not None:
            if hasattr(base, "scales") and hasattr(base, "group_size"):
                out_dim = base.scales.shape[0]
                in_dim = base.scales.shape[1] * base.group_size
            else:
                out_dim, in_dim = base_w.shape[0], base_w.shape[1]
            if A.shape[1] != in_dim or B.shape[0] != out_dim:
                raise ValueError(
                    f"[H3-LoRA] shape mismatch at {mod_path}: "
                    f"base is [{out_dim},{in_dim}] but LoRA "
                    f"A={tuple(A.shape)} B={tuple(B.shape)}"
                )

        overlay = _LoRAOverlay(base, A, B, alpha)
        setattr(parent, leaf, overlay)
        wrapped += 1

        if "final_layer" in mod_path:
            counts["final"] += 1
        elif "token_refiner" in mod_path:
            counts["refiner"] += 1
        elif ".attn." in mod_path:
            counts["attn"] += 1
        elif ".mlp." in mod_path:
            counts["mlp"] += 1
        elif "adaln" in mod_path:
            counts["adaln"] += 1
        else:
            counts["other"] += 1

    if verbose:
        print(f"[H3-LoRA] wrapped {wrapped} modules in {time.time()-t0:.2f}s "
              f"(alpha={alpha}) - attn={counts['attn']}, mlp={counts['mlp']}, "
              f"adaln={counts['adaln']}, final={counts['final']}, "
              f"refiner={counts['refiner']}, other={counts['other']}")

    del tensors
    return wrapped
