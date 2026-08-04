# MiniMax H3 — MLX port

Joint audio+video DiT that packs text, references, audio, and video into a
single sequence and denoises them together. 50-layer transformer over 5376
hidden dim, 24-ch video VAE (16× spatial / 4× temporal) + 32-ch stereo audio
VAE at 40 latent fps, Qwen3-VL-32B (layer-50 truncation) conditioning.

## Port status

**Phases 1–3 of 8 complete.** Video VAE + Audio VAE both encode/decode with
numerical parity to the PyTorch reference (>40 dB end-to-end). See
[`PORT_PLAN.md`](./PORT_PLAN.md) for the full plan and
[`VIDEO_VAE_NOTES.md`](./VIDEO_VAE_NOTES.md) / [`AUDIO_VAE_NOTES.md`](./AUDIO_VAE_NOTES.md)
for architecture notes.

| Phase | Scope | Effort | Status |
|---|---|---|---|
| 1 | Architecture recon + scaffold + config | 1–2 h | ✅ Done |
| 2 | Video VAE (3D causal CNN + ViT3D) | ~1 week est. → **1 day actual** | ✅ Done |
| 3 | Audio VAE (DAC + BigVGAN, 32 kHz stereo) | 3–5 days est. → **~1 hour actual** | ✅ Done |
| 4 | DiT transformer (packed-token, 3-axis RoPE) | 2–3 weeks | ⏳ Not started |
| 5 | Flow-matching scheduler (dual sigma shift) | 2–3 days | ⏳ Not started |
| 6 | Ref2VA safetensors → MLX conversion | 3–5 days | 🚧 Video VAE portion done |
| 7 | Pipeline glue + smoke test | 3–5 days | ⏳ Not started |
| 8 | 4-bit quantization + benchmark | 3–5 days | ⏳ Not started |

Estimated total: **4–6 weeks** of focused work; running ahead of schedule.

### Phase 2 evidence

- Encoder parity (64×64×1f): **139.25 dB PSNR** vs PyTorch reference
- Decoder parity (4×4×1t):   **62.99 dB PSNR** vs PyTorch reference
- Full round-trip on face:   **63.05 dB** MLX-vs-PT (both models produce the
  same ~15 dB reconstruction — the VAE is designed for 17-frame clips, not
  images)
- 5 frames × 384×384 encode + decode: **1.88 s + 0.25 s** (bf16, cold)
- Peak resident set: **5.32 GB** (weights 5.0 + activations ~0.3)

Run the tests:
```
python -m pytest tests/test_h3_video_vae.py -v -s
```

### Phase 3 evidence

- Encoder parity (1 s stereo synthetic): **63.20 dB PSNR** vs PyTorch reference
- Decoder-only parity (PT z → MLX decoder): **43.74 dB PSNR**
- End-to-end MLX vs PyTorch: **42.85 dB PSNR** (peak=2 audio range)
- Real-audio round-trip (Dr. Wang seg_003, 5.48 s Mandarin speech):
  **36.56 dB PSNR**, Whisper transcript **100% character-match** (identical
  Chinese output between original and round-trip)
- 5 s stereo @ 32 kHz encode + decode: **99 ms + 526 ms** (bf16 storage, MLX
  Metal) — **9.5× realtime**
- Peak resident set: **785 MB** (~289 MB weights + activations)
- Peak unified-memory footprint: **~16 GB** (MLX arena)

Run the tests:
```
python -m pytest tests/test_h3_audio_vae.py -v -s
```

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
- `VIDEO_VAE_NOTES.md` — architecture doc for the Video VAE (Phase 2)
- `video_vae.py` — **implemented** (Phase 2): 609 LOC, encode/decode
- `convert.py` — Video VAE + Audio VAE converters implemented
- `attention.py`, `blocks.py`, `model.py`, `packed_layout.py`, `rope.py` — DiT (Phase 4, stubs)
- `audio_vae.py` — **implemented** (Phase 3): 644 LOC, `MiniMaxH3AudioVAE.encode`/`decode`
- `AUDIO_VAE_NOTES.md` — architecture doc for the Audio VAE (Phase 3)
- `scheduler.py` — Phase 5 stub
- `text_encoder_bridge.py` — Phase 6 stub
- `pipeline.py`, `generate.py` — Phase 7 stubs

Remaining stubs raise `NotImplementedError("Phase N: ...")` and carry a `TODO`
line pointing at the exact ComfyUI file + line range to reference when built.
