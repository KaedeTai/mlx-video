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

---

# Phase 8.6 — "Frosted glass" (identity + sharpness) fix

Reported symptom: 33-frame 384x384 Ref2VA samples (bf16 pipeline, Q4 text
encoder) generated at 15 steps looked like Dr. Wang was behind frosted glass.
Visual inspection actually revealed **two** compounding failures:

1. **Wrong identity, not just blur.** The mid-frame of the 15-step sample
   showed a completely different person (Caucasian, brown hair, office
   background) — the ref image was being catastrophically mis-encoded.
2. **Under-sampled tail.** With `shift_video=12`, the flow-matching schedule
   concentrates sigmas near 1.0 and lands the last Euler step from
   `sigma=0.46 → 1e-5` at 15 steps. That is a huge single Euler step over
   a highly non-linear region and softens high-frequency detail.

## Root cause 1: double imagenet normalization in generate.py

`video_vae.encode` expects **NCDHW input in [-1, 1]** — internally it does
`(x + 1) * 0.5` to shift back to `[0, 1]` then applies the imagenet `(x -
mean) / std` once (matches the reference `VAEProcessor.transform_tensor`).

`generate.py --ref-image` pre-normalized with imagenet then handed the
result to `video_vae.encode`, so the encoder saw an already-normalized
tensor pretending to be pixel-space and re-normalized it. For a mid-gray
(0.5) pixel this shifted the input from the correct `[0.066, 0.196,
0.418]` to `[0.209, 0.635, 1.346]` — the encoder went far out of
distribution on the ref image, producing latents that decoded to a
different face. Fix: `arr = arr * 2 - 1` (a single `[0,1] -> [-1,1]`
map), matching the wrapper contract.

## Root cause 2: default `num_steps=15` too few

`MiniMaxH3Scheduler.set_timesteps(N)` with `shift_video=12` produces
`sigmas[N-1]` values of ~0.46 (N=15), 0.29 (N=30), 0.20 (N=50); the last
Euler step then integrates that entire delta in one go. Raising the CLI
default to 30 (matching `~/models/MiniMax-H3-raw/run_ref2va_demo.py`
`DEFAULT_STEPS = 30`) reduces the final delta by ~40% and further steps
help continue to N≈50 with diminishing return.

Reference does **not** use CFG (grep of `~/models/MiniMax-H3-raw/` for
`guidance_scale` / `do_classifier_free_guidance` / `negative_prompt`
returns no hits); H3 is guidance-distilled. VAE `latents_mean/std` and
`imagenet` pixel mean/std constants match the reference config exactly.
Scheduler `sigmas[-1] = 1e-5` (not stuck at high sigma). So neither CFG
nor a wrong latent scale nor a schedule-tail bug are the culprit — the
generate.py double-normalize is.

## Verification

Ref image sharpness (Laplacian variance, edge dx/dy, per-channel mean RGB):

| Sample                              | lap_var | dx    | dy    | mean  | per-channel        |
|-------------------------------------|--------:|------:|------:|------:|--------------------|
| REF (original 240x157)              |   414.5 |  6.90 |  5.53 | 131.7 | 150 / 126 / 119    |
| REF resized 384x384                 |    40.9 |  2.85 |  3.46 | 131.7 | 150 / 126 / 119    |
| VAE roundtrip (correct [-1,1])      |   158.8 |  3.60 |  4.24 |  95.9 | 118 /  90 /  79    |
| **BASELINE (buggy 15 steps)**       |   301.0 |  4.41 |  3.74 | 127.3 | 130 / 126 / 126    |
| A15 normfix                         |   172.3 |  3.58 |  4.20 | 150.2 | 166 / 146 / 139    |
| A30 normfix                         |   205.0 |  3.90 |  4.60 | 153.4 | 169 / 149 / 142    |
| A30 dpmpp2m                         |   175.6 |  3.65 |  4.28 | 149.9 | 166 / 145 / 138    |
| **A50 normfix (sharp target)**      |   241.0 |  4.20 |  4.87 | 155.5 | 172 / 151 / 144    |

The baseline `lap_var=301` was inflated by high-frequency **artifacts**
(the wrong-face + hash-pattern texture) rather than real detail — its
per-channel RGB is nearly neutral (~130/126/126) despite the warm-skin
reference (~150/126/119). All post-fix runs match the reference color
signature (R > G > B, warm tone), confirming identity is now correctly
transferred.

Recommended production config: `--num-steps 30` for interactive iteration,
`--num-steps 50` for the final delivery. Euler is fine; the current
DPM++ 2M implementation trapezoidally averages consecutive Euler slopes
and matched Euler-30 in this test, so there's no reason to switch.

## Files touched

- `mlx_video/models/minimax_h3/generate.py`
  - `--ref-image` path: correct `[0,1] -> [-1,1]` (was double imagenet).
  - Default `--num-steps 15 -> 30`.
  - Added `--sampler {euler,dpmpp_2m}` and `--shift-video FLOAT` overrides.
