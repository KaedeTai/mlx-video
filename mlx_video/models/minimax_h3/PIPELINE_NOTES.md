# H3 Pipeline — MLX port notes (Phase 7)

## Files

| File | LOC | Notes |
|---|---|---|
| `pipeline.py` | 250 | Geometry helpers + `H3Pipeline` container + `.generate()` |
| `generate.py` | 140 | CLI + ffmpeg mux |
| `text_encoder_bridge.py` | 60 | `DummyTextEncoder` (Phase-8 will add Qwen3-VL-32B truncation) |

## Smoke test results (M-series, bf16)

Command:
```
python -m mlx_video.models.minimax_h3.generate \
  --length 33 --width 384 --height 384 --num-steps 15 \
  --ref-image ~/movie/wang_wenchin/faces/0100.jpg \
  --ref-audio /tmp/h3_smoke_silence_3s.wav \
  --prompt "A person speaking to camera warmly, professional setting" \
  --output ~/tmp/h3_mlx_smoke_test.mp4
```

Result: `72.6s wall time`, `100 KB` mp4 (H264 + AAC).

Per-step timing (seq_len 2258, includes text=8 + ref_img=144 + ref_audio=240 + audio=130 + video=1728):
```
step  1/15: sigma=1.0000, step=6.3s   (first — warm-up)
step  2..8: sigma=0.99..0.93, step=3.8–4.2s
step  9..15: sigma=0.91..0.46, step=4.8–5.4s
```
Mean ≈ 4.7 s/step.

Full-config decode:
- Video VAE decode (12 latent frames → 48 pixel frames × 384×384): ~2 s
- Audio VAE decode (65 latent frames → 52,000 samples stereo): ~1 s
- ffmpeg mux: <1 s

## What the smoke test verifies

- ✅ Full 33 B DiT loads and forwards without shape/dtype errors
- ✅ PackedLayout builds the correct `[text | ref_img | ref_audio | audio | video]`
  segment order and `seq_len` matches expectation
- ✅ RoPE table (2258 × 96 pair angles) computes fine
- ✅ 50 DiT blocks × 15 steps run to completion (no NaN, no OOM)
- ✅ Video VAE decode produces `[1, 3, 48, 384, 384]` frames in `[-1, 1]`
- ✅ Audio VAE decode produces `[1, 2, 52000]` stereo waveform
- ✅ ffmpeg accepts the frames + waveform and writes a playable mp4

## Known limitations (Phase 7 → Phase 8 backlog)

- **Text encoder is a `DummyTextEncoder`** returning random ~0.01-magnitude
  embeddings. Output video therefore has no semantic content — smoke test
  measures plumbing, not quality. Frame 1 sanity: mean ≈ 90, std ≈ 3 per
  channel — low-contrast noise field, as expected for random text
  conditioning.
- **DPM++ 2M sampler not exercised** — smoke used Euler.
- **No spatial tiling on VAE decode** — fine at 384×384; larger canvases
  (768×1344 with 124 frames) may exceed unified memory.
- **fp32-island casting on adaln_proj weights** — currently converted as bf16
  since they're bf16 in the source; may hurt numerical accuracy in later
  passes if any layers were meant to be fp32 at inference.
- **Peak RSS report (60 GB → 63 GB)** is macOS mmap virtual-page accounting;
  actual resident memory ≈ 30-40 GB (weights are lazy-materialized).
