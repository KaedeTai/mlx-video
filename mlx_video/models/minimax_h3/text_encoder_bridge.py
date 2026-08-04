"""Qwen3-VL-32B truncated-to-layer-50 conditioning for MiniMax H3.

Port target: comfy/text_encoders/minimax.py (201 LOC, whole file).

Key mechanics:
- Not chat-templated; raw token IDs with <Picture i> / <Video k> / <Audio j>
  labels plus explicit <|vision_start|> / <|vision_end|> spans.
- Task-specific presentation order: images then videos (with audio labels
  before their video), then standalone audio; per-type 1-based ordinals.
- Returns unnormalized layer-50 hidden state (no final norm) as conditioning.
- Also returns `minimax_token_tags` (0 = video-modality, 1 = text-modality)
  for the DiT's per-tag adaLN routing.

Weight-source options (decision deferred to Phase 6):
  (a) reuse ~/models/Qwen3-VL-32B-Instruct-4bit (mlx-community) — wrap and
      stop propagation at layer 50.
  (b) convert Ref2VA/text_encoder shards (already truncated to layer 50).
"""

# TODO: reference from /tmp/h3_recon/ComfyUI/comfy/text_encoders/minimax.py:1-201


raise NotImplementedError("Phase 6: wire Qwen3-VL layer-50 truncation")
