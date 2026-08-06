"""H3 v16 Sub2: build a stripped deployment checkpoint.

Given a full HEAVY_SUFFIX Q4 build at ``--src`` (e.g. ``MiniMaxH3-Ref2VA-MLX-Q4``),
produce a stripped copy at ``--dst`` (default suffix ``-Stripped``) where the
per-block ``adaln_proj.linear`` weights have been dropped (they live at ~26 GiB
in bf16 for the 50-block bank) and a precomputed step-indexed AdaLN cache is
persisted next to the model. The bundle carries a JSON signature so the runtime
loader can refuse a mismatch (model / LoRA / schedule / conditioning drift).

Contents of ``<dst>/dit/``::

    model.safetensors        # HEAVY_SUFFIX Q4 with adaln_proj.linear.* dropped
    quantization.json        # copied from src, unchanged
    modulation_cache.npz     # per-step tables (float32)
    cache_signature.json     # CacheSignature — verified at load time

Usage::

    python -m scripts.h3.build_stripped_bundle \\
        --src ~/mlx-video/mlx-models/MiniMaxH3-Ref2VA-MLX-Q4 \\
        --dst ~/mlx-video/mlx-models/MiniMaxH3-Ref2VA-MLX-Q4-Stripped \\
        --num-steps 30 --has-visual-cond --has-audio-cond

Skip ``--turbo-lora`` unless the deployment plan is Turbo — otherwise the cache
will only be valid at that exact LoRA scale.

Heavy: loading the full DiT + building the cache peaks around ~35-45 GiB.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx.utils import tree_flatten


def _log(msg: str):
    print(f"[strip {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _drop_adaln_from_flat(flat: dict) -> tuple[dict, int]:
    """Filter out any parameter whose key contains ``.adaln_proj.linear.``.

    Returns ``(filtered_dict, bytes_freed)``.
    """
    keep: dict = {}
    freed = 0
    for k, v in flat.items():
        if ".adaln_proj.linear." in k or k.startswith("final_layer.adaln_proj.linear."):
            freed += int(v.size * v.dtype.size) if hasattr(v, "size") else 0
            continue
        keep[k] = v
    return keep, freed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source Q4 bundle root")
    ap.add_argument("--dst", required=True, help="destination stripped bundle root")
    ap.add_argument("--num-steps", type=int, required=True)
    ap.add_argument("--has-visual-cond", action="store_true")
    ap.add_argument("--has-audio-cond", action="store_true")
    ap.add_argument("--turbo-lora", default=None,
                    help="Path to Turbo LoRA safetensors — cache will be built with LoRA "
                         "applied, so the resulting bundle is only valid for Turbo runs at "
                         "the same alpha.")
    ap.add_argument("--turbo-lora-alpha", type=float, default=1.0)
    ap.add_argument("--tag", default="",
                    help="Freeform tag stored in the signature (e.g. run id / date).")
    ap.add_argument("--skip-verify", action="store_true",
                    help="Skip verify_cache_bitwise (only use if RAM contested).")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    dst = Path(args.dst).expanduser()
    src_dit = src / "dit"
    dst_dit = dst / "dit"
    dst_dit.mkdir(parents=True, exist_ok=True)

    src_safe = src_dit / "model.safetensors"
    src_qjson = src_dit / "quantization.json"
    if not src_safe.exists():
        raise SystemExit(f"missing {src_safe}")
    if not src_qjson.exists():
        raise SystemExit(f"missing {src_qjson}")

    # ---- Load full pipeline into memory (heavy) ----------------------
    _log(f"loading pipeline from {src} ...")
    t0 = time.time()
    from mlx_video.models.minimax_h3.pipeline import load_pipeline
    pipe = load_pipeline(src)
    _log(f"loaded in {time.time()-t0:.1f}s")

    lora_hash: Optional[str] = None
    if args.turbo_lora:
        from mlx_video.models.minimax_h3.lora import load_turbo_lora
        from mlx_video.models.minimax_h3.modulation_cache import hash_file
        _log(f"applying Turbo LoRA {args.turbo_lora} @ alpha={args.turbo_lora_alpha}")
        load_turbo_lora(pipe.dit, args.turbo_lora, alpha=args.turbo_lora_alpha, verbose=True)
        lora_hash = hash_file(args.turbo_lora, max_bytes=64 * 1024**2)
        lora_hash = f"{lora_hash}:alpha={args.turbo_lora_alpha}"

    # ---- Compute model hash + build cache ----------------------------
    from mlx_video.models.minimax_h3.modulation_cache import (
        DEFAULT_SCHEDULE_NAME, ModulationCache, _live_signature_from_pipeline,
        hash_file, per_step_unique_t, save_cache_bundle, verify_cache_bitwise,
    )
    _log("hashing source safetensors (leading 64 MiB) ...")
    model_hash = hash_file(src_safe, max_bytes=64 * 1024**2)
    _log(f"model_hash={model_hash[:16]}...")

    pipe.scheduler.set_timesteps(args.num_steps)
    sigmas_list = pipe.scheduler.sigmas.tolist()
    step_ut = per_step_unique_t(
        sigmas_list,
        has_visual_cond=args.has_visual_cond,
        has_audio_cond=args.has_audio_cond,
        shift_video=pipe.scheduler.shift_video,
        shift_audio=pipe.scheduler.shift_audio,
    )
    signature = _live_signature_from_pipeline(
        num_steps=args.num_steps,
        num_blocks=len(pipe.dit.blocks),
        has_visual_cond=args.has_visual_cond,
        has_audio_cond=args.has_audio_cond,
        shift_video=pipe.scheduler.shift_video,
        shift_audio=pipe.scheduler.shift_audio,
        sigma_min=1e-5,
        schedule_name=DEFAULT_SCHEDULE_NAME,
        dtype="float32",
        use_adaln_curves=bool(getattr(pipe.dit, "use_adaln_curves", False)),
        cache_final=True,
        model_hash=model_hash,
        lora_hash=lora_hash,
        layout_hash="unbound",  # stripped bundle is layout-agnostic
        per_step_ut=step_ut,
        tag=args.tag,
    )

    _log(f"building step-indexed cache: {len(step_ut)} steps M={[len(x) for x in step_ut]}")
    t0 = time.time()
    cache = ModulationCache.build(pipe.dit, step_ut, signature, dtype=mx.float32, cache_final=True)
    _log(f"cache built in {time.time()-t0:.1f}s, size={cache.nbytes()/1024**2:.1f} MiB")

    if not args.skip_verify:
        _log("bitwise verify against live adaln_proj ...")
        t0 = time.time()
        n_checks, max_diff = verify_cache_bitwise(cache, pipe.dit, verbose=True)
        _log(f"verify OK in {time.time()-t0:.1f}s: {n_checks} tuple comparisons, max_diff={max_diff}")
        if max_diff != 0.0:
            raise SystemExit(f"[strip] verify failed: max_diff={max_diff}")
    else:
        _log("SKIPPED verify_cache_bitwise (--skip-verify)")

    # ---- Persist stripped safetensors --------------------------------
    _log(f"reading {src_safe} into memory for strip ...")
    t0 = time.time()
    src_flat = mx.load(str(src_safe))
    _log(f"read {len(src_flat)} tensors in {time.time()-t0:.1f}s")

    stripped, freed = _drop_adaln_from_flat(src_flat)
    _log(f"stripped adaln_proj.linear.*: dropped {len(src_flat)-len(stripped)} tensors, "
         f"~{freed/1024**3:.2f} GiB")

    dst_safe = dst_dit / "model.safetensors"
    _log(f"writing {dst_safe} ...")
    t0 = time.time()
    mx.save_safetensors(str(dst_safe), stripped)
    _log(f"wrote {dst_safe.stat().st_size/1024**3:.2f} GiB in {time.time()-t0:.1f}s")

    # ---- Copy quantization.json + note stripped fields ---------------
    src_meta = json.loads(src_qjson.read_text())
    src_meta["stripped_adaln_proj_linear"] = True
    src_meta["v16_bundle"] = True
    (dst_dit / "quantization.json").write_text(json.dumps(src_meta, indent=2))
    _log("wrote quantization.json")

    # ---- Persist cache + signature ----------------------------------
    save_cache_bundle(
        cache,
        dst_dit / "modulation_cache.npz",
        dst_dit / "cache_signature.json",
    )
    _log(f"wrote modulation_cache.npz + cache_signature.json")

    # ---- Symlink VAEs so downstream code just works -----------------
    for sub in ("video_vae", "audio_vae"):
        src_link = src / sub
        dst_link = dst / sub
        if dst_link.exists() or dst_link.is_symlink():
            continue
        try:
            dst_link.symlink_to(src_link.resolve())
            _log(f"symlinked {sub} -> {src_link.resolve()}")
        except OSError as exc:
            _log(f"failed to symlink {sub}: {exc}")

    _log(f"DONE. Stripped bundle at {dst}")


if __name__ == "__main__":
    main()
