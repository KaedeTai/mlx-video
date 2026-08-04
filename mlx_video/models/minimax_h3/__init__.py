"""MiniMax H3 joint audio+video DiT — MLX port.

Port progress:
- Phase 1  DONE  — architecture recon + scaffold + config.py
- Phase 2  DONE  — Video VAE (3D causal CNN encoder + ViT3D decoder)
- Phase 3  DONE  — Audio VAE (DAC encoder + BigVGAN decoder, 32 kHz stereo)
- Phase 4  DONE  — DiT transformer (50 layers, packed-token, 3-axis RoPE)
- Phase 5  ....  — Flow-matching scheduler with dual sigma shift
- Phase 6  ....  — Ref2VA safetensors → MLX weight conversion
- Phase 7  ....  — Pipeline glue + smoke test

See PORT_PLAN.md for the full plan and reference-implementation checklist.
"""

from mlx_video.models.minimax_h3.config import MiniMaxH3Config, load_ref2va_config
from mlx_video.models.minimax_h3.model import MiniMaxH3Model, time_shift_sigma, time_shift_slope
from mlx_video.models.minimax_h3.packed_layout import PackedLayout, RefBlock, Keyframe

__all__ = [
    "MiniMaxH3Config",
    "MiniMaxH3Model",
    "PackedLayout",
    "RefBlock",
    "Keyframe",
    "load_ref2va_config",
    "time_shift_sigma",
    "time_shift_slope",
]
