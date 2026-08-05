# H3 MLX port — first production-quality sample run (v1)

Ran on branch `minimax-h3-port` at commit `a0ddde3a` (Phase 8.9-c: full
vision-token splicing).  See PORT_PLAN.md for the phase notes; this file
records what the first "run it like a product" pass produced.

## Setup

- ref image: `~/movie/wang_wenchin/faces/0019.jpg`  (41 kB portrait crop, Dr Wang / Willy)
- ref audio: `~/tmp/willy_audio_for_h3_sample.wav` (32 kHz mono, 6.0 s clean Willy vocal)
  - Fallback B: Higgsfield session was expired so used the already-captured Willy
    sample from `~/tmp/drwang_clean_6s_32k.wav` instead of generating fresh TTS.
- text encoder: `~/mlx-video/mlx-models/H3-TextEncoder-MLX-bf16`
  (Comfy-Org H3 truncated Qwen3-VL-32B, 50 layers, 51 GB bf16)
- DiT: `~/mlx-video/mlx-models/MiniMaxH3-Ref2VA-MLX-Q4`
- prompt: `王博士對著鏡頭溫暖地說話，專業辦公室背景，自然打光，微笑，微微點頭`
- seed: 42
- length: 49 frames requested -> snaps to 56 (17*3+5)
- num-steps: 30
- sampler: euler (default), shift_video 12.0

Driver: `python -m scripts.h3.sample_phase89b` (RAM plan B — encode
text with 51 GB encoder, save context, free encoder, then load DiT+VAEs
and inject saved context).

## Results

### 384x576 (h3_production_v1_sample.mp4, 105 KB)

| metric               | value |
|----------------------|-------|
| wall time            | 316.4 s (5m16s) |
| per-step             | ~10.4 s |
| peak Python RSS      | 42.15 GB (all in encoder stage; DiT stage ~30 GB) |
| seq_len              | 4592 |
| output duration      | 2.33 s (56 frames @ 24 fps) |
| audio                | 32 kHz stereo, 2.325 s |

Mid-frame identity intact. No 16-px grid. No gray flatness.
Whisper transcript (small model, zh): `並義特利社格過獎` (unintelligible).

### 512x768 (h3_production_v1_hires.mp4, 198 KB)

| metric               | value |
|----------------------|-------|
| wall time            | 533.8 s (8m54s) |
| per-step             | ~17.6 s |
| peak Python RSS      | 42.50 GB |
| seq_len              | 7616 |
| output duration      | 2.33 s (56 frames @ 24 fps) |
| audio                | 32 kHz stereo, 2.325 s |

Dramatically sharper than 384x576. Warm smile matches prompt. Fluorescent
office ceiling emerges in background. Striped tie + blazer visible.
Whisper transcript (small model, zh): `我們下集見 敬禮託國慶`
(off-prompt but recognizably Chinese phonemes; timbre matches Willy).

### 768x1024 — skipped

Killed after 9 min at step 0. System is a 128 GB M-series; swap was
already at 36/37 GB when the DiT tried to allocate step-1 attention
buffers for a ~30 500-token sequence. 1.8 % CPU with 50 MB/s sustained
disk I/O — swap-thrashing, would have taken 40-60 min per step.

To reach native res on this box we need either
  a) SDPA-tiled attention (chunk sequence dim) — Wan S2V port does this,
  b) further weight-only Q for encoder (bf16 -> Q8/Q4), or
  c) offload DiT layers to disk between forward passes.

## vs. Wan S2V production config (memory)

Wan S2V ships a similar pattern (encode+free, then generate) and hits
native res on the same 128 GB box because its text encoder is 5-10x
smaller than the 51 GB H3 encoder. H3's Qwen3-VL-32B mid-layer
requirement is the current ceiling on native-res H3 — parity with Wan
S2V would need either encoder quantization or a stream/spill loader.

## Notes on audio intelligibility

Both samples' Whisper transcripts are off-prompt. Two contributing factors:
1. The prompt describes the *scene*, not the intended speech content —
   the model has to invent what to say. To get intelligible Chinese
   matching a target sentence, put the target sentence *in the prompt*
   as direct quotation ("王博士溫暖地說：大家好，我是王文欽博士。").
2. 2.3 s is very short for Whisper to lock context; longer clips
   (length=105, ~4.4 s at 24 fps) transcribe better.

## Reproduce

```
cd ~/mlx-video
python -m scripts.h3.sample_phase89b \
  --prompt "王博士對著鏡頭溫暖地說話，專業辦公室背景，自然打光，微笑，微微點頭" \
  --ref-image ~/movie/wang_wenchin/faces/0019.jpg \
  --ref-audio ~/tmp/willy_audio_for_h3_sample.wav \
  --encoder-path ~/mlx-video/mlx-models/H3-TextEncoder-MLX-bf16 \
  --model-root ~/mlx-video/mlx-models/MiniMaxH3-Ref2VA-MLX-Q4 \
  --width 512 --height 768 --length 49 --num-steps 30 --seed 42 \
  --output ~/tmp/h3_production_v1_hires.mp4
```
