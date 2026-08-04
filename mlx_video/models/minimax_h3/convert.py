"""Ref2VA → MLX weight conversion for MiniMax H3.

Usage (Phase 6):
    python -m mlx_video.models.minimax_h3.convert \\
        --src ~/models/MiniMax-H3-raw/Ref2VA \\
        --dst ~/models/MiniMax-H3-Ref2VA-mlx

Four sub-converters:
  1. convert_transformer  — 13 bf16 shards → 1 fp16 MLX safetensors. Keys map
                            1:1 to our MiniMaxH3Model (see PORT_PLAN.md §4).
  2. convert_video_vae    — single-file, no rewrite required.
  3. convert_audio_vae    — FOLD weight-norm (weight_g * weight_v/||weight_v||)
                            on every conv; drop logs_proj + mask_token.
  4. convert_text_encoder — TRUNCATE Qwen3-VL to layers 0-49; drop lm_head +
                            final norm.

Peak RAM safety: stream one shard at a time via `mx.load` on individual files.
Skeleton pattern: mlx_video/models/wan_2/convert.py.
"""

# TODO: reference from ~/mlx-video/mlx_video/models/wan_2/convert.py (skeleton)
# TODO: reference from /tmp/h3_recon/ComfyUI/comfy/sd.py (VAE detection keys)


raise NotImplementedError("Phase 6: implement Ref2VA → MLX weight conversion")
