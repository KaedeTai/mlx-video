"""H3 DiT building blocks: TimeEmbedder, AdalnProj, TokenRefiner, DiTBlock, FinalLayer.

Port target: comfy/ldm/minimax/model.py (lines 114-279).

Key notes:
- TimeEmbedder: sin/cos on [M] in [0, 1], **cos before sin**, fp32.
- AdalnProj: silu → linear → view [M*modalities, expand*hidden] → chunk.
- Modulation is segment-indexed (`mod_segments = [(a, b, row)]`) — translate
  ComfyUI's in-place `_mod_scale_shift` / `_mod_gate` to MLX gather/scatter.
- FinalLayer video_out / audio_out are fp32 islands (kept at torch.float32 in
  the checkpoint) — mirror that in `convert.py`.
"""

# TODO: reference from /tmp/h3_recon/ComfyUI/comfy/ldm/minimax/model.py:114-279


raise NotImplementedError("Phase 4: implement H3 blocks")
