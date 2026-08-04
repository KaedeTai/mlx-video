"""MiniMax H3 joint audio+video DiT — MLX port.

Port progress (Phase 1 of 8 complete):
- Phase 1  DONE  — architecture recon + scaffold + config.py
- Phase 2  TODO  — Video VAE (3D causal CNN encoder + ViT3D decoder)
- Phase 3  TODO  — Audio VAE (DAC encoder + BigVGAN decoder, 32 kHz stereo)
- Phase 4  TODO  — DiT transformer (50 layers, packed-token, 3-axis RoPE)
- Phase 5  TODO  — Flow-matching scheduler with dual sigma shift
- Phase 6  TODO  — Ref2VA safetensors → MLX weight conversion
- Phase 7  TODO  — Pipeline glue + smoke test
- Phase 8  TODO  — 4-bit quantization + benchmark

See PORT_PLAN.md for the full plan and reference-implementation checklist.
"""

from mlx_video.models.minimax_h3.config import MiniMaxH3Config, load_ref2va_config

# Phase-4 exports (populated once model.py / video_vae.py / audio_vae.py land)
# from mlx_video.models.minimax_h3.model import MiniMaxH3Model
# from mlx_video.models.minimax_h3.video_vae import MiniMaxH3VideoVAE
# from mlx_video.models.minimax_h3.audio_vae import MiniMaxH3AudioVAE

__all__ = [
    "MiniMaxH3Config",
    "load_ref2va_config",
]
