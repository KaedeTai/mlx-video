# MiniMax H3 — MLX port

**Status:** end-to-end pipeline runs on Apple Silicon. Produces a joint
video+audio mp4 from prompt + optional reference image + optional reference
audio. See `PORT_PLAN.md` for the full plan and per-phase completion notes.

## What's in this directory

| File | Purpose |
|---|---|
| `config.py` | `MiniMaxH3Config` dataclass (5376-dim / 50-layer / 56-head DiT) |
| `video_vae.py` | 3D causal CNN encoder + 36-block ViT3D decoder (Phase 2) |
| `audio_vae.py` | DAC encoder + BigVGAN decoder, 32 kHz stereo, 40 latent fps (Phase 3) |
| `rope.py` | 3-axis split-half rotary embedding (Phase 4) |
| `attention.py` | H3 attention (qkv-fused + per-head RMSNorm + split-half rope) |
| `blocks.py` | TimeEmbedder / AdalnProj / MLP / RefinerBlock / DiTBlock / FinalLayer |
| `packed_layout.py` | `[text \| cond \| audio \| video]` PackedLayout builder |
| `model.py` | `MiniMaxH3Model` (33 B params, batch=1, packed-token DiT) |
| `scheduler.py` | Dual-shift flow-matching scheduler (video drives; audio derived) |
| `convert.py` | Ref2VA safetensors → MLX safetensors converter |
| `text_encoder_bridge.py` | `DummyTextEncoder` (smoke) + `TextEncoderBridge` (real Qwen3-VL Q4, truncated at layer 50) |
| `pipeline.py` | `H3Pipeline` container + `.generate()` |
| `generate.py` | CLI entry (ffmpeg mux to mp4) |

## Weight conversion

```bash
# All at once (DiT + Video VAE + Audio VAE)
python -m mlx_video.models.minimax_h3.convert --component all --dtype bf16

# Individual components
python -m mlx_video.models.minimax_h3.convert --component dit --dtype bf16
python -m mlx_video.models.minimax_h3.convert --component video-vae --dtype bf16
python -m mlx_video.models.minimax_h3.convert --component audio-vae --dtype bf16
```

Outputs go to `~/mlx-video/mlx-models/MiniMaxH3-Ref2VA-MLX-bf16/{dit,video_vae,audio_vae}/model.safetensors`.

## End-to-end generation

```bash
# Smoke test (5 frames, 3 steps, ~20 s)
python -m mlx_video.models.minimax_h3.generate \
  --length 5 --width 384 --height 384 --num-steps 3 \
  --output ~/tmp/h3_smoke.mp4

# Full smoke test (33 frames + ref image + silent ref audio, 15 steps, ~75 s)
python -m mlx_video.models.minimax_h3.generate \
  --length 33 --width 384 --height 384 --num-steps 15 \
  --ref-image ~/movie/wang_wenchin/faces/0100.jpg \
  --ref-audio /tmp/h3_smoke_silence_3s.wav \
  --prompt "A person speaking to camera warmly, professional setting" \
  --output ~/tmp/h3_mlx_smoke_test.mp4
```

The pipeline defaults to a **`DummyTextEncoder`** for fast smoke tests. For
real generation, pass the Qwen3-VL-32B-Instruct-4bit MLX checkpoint:

```bash
python -m mlx_video.models.minimax_h3.generate \
  --text-encoder-path ~/models/Qwen3-VL-32B-Instruct-4bit \
  --model-root ~/mlx-video/mlx-models/MiniMaxH3-Ref2VA-MLX-Q4 \
  --length 33 --width 384 --height 384 --num-steps 15 \
  --ref-image ~/movie/wang_wenchin/faces/0100.jpg \
  --prompt "The man is speaking warmly to the camera..." \
  --output ~/tmp/h3.mp4
```

Point `--model-root` at the `MLX-Q4` directory to get the 4-bit DiT
(35 GB vs 62 GB bf16, ~50 GB peak RSS at 33 f x 384^2 vs ~90 GB bf16).
Both bf16 and Q4 checkpoints load transparently -- the loader detects
`dit/quantization.json` and re-applies the same predicate before
`load_weights`.

## Tests

```bash
python -m tests.test_h3_video_vae     # Phase 2 — 4/4
python -m tests.test_h3_audio_vae     # Phase 3 — 5/5
python -m tests.test_h3_dit_smoke     # Phase 4 — 3/3
python -m tests.test_h3_scheduler     # Phase 5 — 5/5
python -m tests.test_h3_dit_load      # Phase 6 — full-weight load
python -m pytest tests/test_h3_text_encoder.py  # Phase 8-1 — 3/3
```

## Phase progress

| Phase | Description | Status |
|---|---|---|
| 1 | Architecture recon + scaffold | ✅ done |
| 2 | Video VAE port | ✅ done (63 dB round-trip PSNR) |
| 3 | Audio VAE port | ✅ done (43 dB PSNR, 9.5× realtime, Whisper 100 % match) |
| 4 | DiT transformer port | ✅ done (33 B params, weight-naming 1:1) |
| 5 | Dual-schedule scheduler | ✅ done |
| 6 | Full weight converter | ✅ done (63 GB bf16, 40.7 s) |
| 7 | Pipeline glue + smoke test | ✅ done (72.6 s / 33 frames × 15 steps) |
| 8 | Real Qwen3-VL text encoder + 4-bit quant + full-res bench | ✅ done |
| 9 | KV-cache for ref blocks; attention tiling at high seq; vision-block Qwen prompt splicing | 📋 planned |
