#!/usr/bin/env bash
# Phase 8.11 sample runner. Produces v1 (spatial tiling) and v3 (with latent
# dump for FFT diagnostic) mp4s + npy dump.
#
# Run from repo root: bash scripts/h3/phase811_samples.sh
set -euo pipefail

REF_IMG="${REF_IMG:-$HOME/movie/wang_wenchin/faces/0100.jpg}"
PROMPT="${PROMPT:-A person speaks warmly at the camera.}"
STEPS="${STEPS:-30}"
LENGTH="${LENGTH:-33}"
WIDTH="${WIDTH:-384}"
HEIGHT="${HEIGHT:-384}"
SEED="${SEED:-0}"
TEXT_ENC="${TEXT_ENC:-$HOME/models/Qwen3-VL-32B-Instruct-4bit}"

mkdir -p "$HOME/tmp"

echo "=== Phase 8.11-1 sample: spatial tiling ON (default post-fix) ==="
python -m mlx_video.models.minimax_h3.generate \
    --prompt "$PROMPT" \
    --width "$WIDTH" --height "$HEIGHT" --length "$LENGTH" \
    --num-steps "$STEPS" --seed "$SEED" \
    --ref-image "$REF_IMG" \
    --text-encoder-path "$TEXT_ENC" \
    --output "$HOME/tmp/h3_phase811_v1_sample.mp4"

echo
echo "=== Phase 8.11-3 sample: dump DiT latent for FFT diagnostic ==="
python -m mlx_video.models.minimax_h3.generate \
    --prompt "$PROMPT" \
    --width "$WIDTH" --height "$HEIGHT" --length "$LENGTH" \
    --num-steps "$STEPS" --seed "$SEED" \
    --ref-image "$REF_IMG" \
    --text-encoder-path "$TEXT_ENC" \
    --dump-latent "$HOME/tmp/h3_phase811_v3_latent.npy" \
    --output "$HOME/tmp/h3_phase811_v3_sample.mp4"

echo
echo "=== FFT report ==="
python scripts/h3/analyze_dit_latent.py \
    "$HOME/tmp/h3_phase811_v3_latent.npy" \
    --pixel-fft "$HOME/tmp/h3_phase811_v3_sample.mp4"

echo
echo "=== Also run the no-tiling baseline for A/B compare ==="
python -m mlx_video.models.minimax_h3.generate \
    --prompt "$PROMPT" \
    --width "$WIDTH" --height "$HEIGHT" --length "$LENGTH" \
    --num-steps "$STEPS" --seed "$SEED" \
    --ref-image "$REF_IMG" \
    --text-encoder-path "$TEXT_ENC" \
    --no-tiling \
    --output "$HOME/tmp/h3_phase811_no_tiling_sample.mp4"

python scripts/h3/analyze_dit_latent.py \
    "$HOME/tmp/h3_phase811_v3_latent.npy" \
    --pixel-fft "$HOME/tmp/h3_phase811_no_tiling_sample.mp4"
