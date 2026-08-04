# MiniMax H3 — MLX port

Joint audio+video DiT that packs text, references, audio, and video into a
single sequence and denoises them together. 50-layer transformer over 5376
hidden dim, 24-ch video VAE (16× spatial / 4× temporal) + 32-ch stereo audio
VAE at 40 latent fps, Qwen3-VL-32B (layer-50 truncation) conditioning.

## Port status

**Phase 1 of 8 complete** — architecture recon + scaffold + config only.
Nothing runs yet. See [`PORT_PLAN.md`](./PORT_PLAN.md) for the full plan.

| Phase | Scope | Effort | Status |
|---|---|---|---|
| 1 | Architecture recon + scaffold + config | 1–2 h | ✅ Done |
| 2 | Video VAE (3D causal CNN + ViT3D) | ~1 week | ⏳ Not started |
| 3 | Audio VAE (DAC + BigVGAN, 32 kHz stereo) | 3–5 days | ⏳ Not started |
| 4 | DiT transformer (packed-token, 3-axis RoPE) | 2–3 weeks | ⏳ Not started |
| 5 | Flow-matching scheduler (dual sigma shift) | 2–3 days | ⏳ Not started |
| 6 | Ref2VA safetensors → MLX conversion | 3–5 days | ⏳ Not started |
| 7 | Pipeline glue + smoke test | 3–5 days | ⏳ Not started |
| 8 | 4-bit quantization + benchmark | 3–5 days | ⏳ Not started |

Estimated total: **4–6 weeks** of focused work.

## Reference implementations

Primary source of truth is ComfyUI's native H3 support (2321 LOC across 5 files):

- `comfy/ldm/minimax/model.py` — DiT transformer (646 LOC)
- `comfy/ldm/minimax/vae.py` — video VAE (694 LOC)
- `comfy/ldm/minimax/audio_vae.py` — audio VAE (443 LOC)
- `comfy/text_encoders/minimax.py` — Qwen3-VL wiring (201 LOC)
- `comfy_extras/nodes_minimax_h3.py` — pipeline recipe (337 LOC)

Weights come from `~/models/MiniMax-H3-raw/Ref2VA/` (134 GB bf16 diffusers).
DiT keys map 1:1 to ComfyUI's module — no rename table needed. Audio VAE needs
weight-norm folding at convert time. See `PORT_PLAN.md` §4 for the mapping
table.

## Usage (planned — Phase 7)

```bash
# Convert Ref2VA once (~15 min on M3-Max)
python -m mlx_video.models.minimax_h3.convert \
    --src ~/models/MiniMax-H3-raw/Ref2VA \
    --dst ~/models/MiniMax-H3-Ref2VA-mlx

# Text-to-video+audio
python -m mlx_video.models.minimax_h3.generate \
    --task t2va \
    --prompt "A cat batting a ball of yarn, camera slowly zooming in." \
    --output cat.mp4

# Reference-to-video+audio
python -m mlx_video.models.minimax_h3.generate \
    --task ref2va \
    --prompt "<Picture 1> walks into the room to <Audio 1>." \
    --ref-image ref_person.png \
    --ref-audio ref_footsteps.wav \
    --output scene.mp4
```

## Files in this package

- `config.py` — `MiniMaxH3Config` dataclass (implemented) + `.from_hf_config()`
- `PORT_PLAN.md` — full port plan with weight mapping and per-phase risks
- `attention.py`, `blocks.py`, `model.py`, `packed_layout.py`, `rope.py` — DiT (Phase 4)
- `video_vae.py` — Phase 2 stub
- `audio_vae.py` — Phase 3 stub
- `scheduler.py` — Phase 5 stub
- `text_encoder_bridge.py` — Phase 6 stub
- `convert.py` — Phase 6 stub
- `pipeline.py`, `generate.py` — Phase 7 stubs

Every stub raises `NotImplementedError("Phase N: ...")` and carries a `TODO` line
pointing at the exact ComfyUI file + line range to reference when it's built.
