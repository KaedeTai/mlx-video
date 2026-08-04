"""Text encoder wiring for MiniMax H3.

MiniMax H3 conditions on the **unnormalized layer-50 hidden state** of
Qwen3-VL-32B (64 layers total, so layer 50 is a mid-network cut). See
``comfy/text_encoders/minimax.py`` in the reference.

There are two viable implementation paths (kept as a design decision):

A) **In-repo Qwen3-VL truncated loader.** Convert Ref2VA's own
   ``text_encoder`` shards (706 language_model keys across 14 shards) to
   MLX, keep layers 0-49 only, drop the vision blocks (~350 keys) and
   ``lm_head`` / ``model.language_model.norm``. Output: single safetensors,
   ~5-7 GB in bf16.

B) **External MLX checkpoint.** Wrap ``mlx-community/Qwen3-VL-32B-Instruct-4bit``
   in a "stop-at-layer-50" adapter. Reuses the ~15 GB q4 file we already have.

The pipeline (Phase 7) supports **either** by expecting a callable
``encode_text(prompt: str) -> mx.array`` of shape ``[1, L, 5120]`` (fp32).

For the Phase 7 smoke test we provide a **DummyTextEncoder** that produces
random or zero-init embeddings of the right shape. The full-quality wiring is
a Phase 8 optimization (video quality without meaningful text conditioning is
clearly limited, but the pipeline plumbing works either way).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np


@dataclass
class DummyTextEncoder:
    """Placeholder text encoder that returns zero-init embeddings.

    Suitable for Phase-7 smoke testing: verifies the pipeline plumbing.
    Video quality will be uncontrolled since the model gets no semantic signal.
    """
    text_dim: int = 5120
    max_len: int = 128
    seed: int = 0

    def encode(self, prompt: str) -> mx.array:
        """Return ``[1, L, text_dim]`` fp32 embeddings.

        Uses a deterministic length proportional to prompt word count (min 4,
        max ``self.max_len``); embeddings are small random values so downstream
        norms don't NaN.
        """
        L = min(max(4, len(prompt.split()) * 2), self.max_len)
        rng = np.random.default_rng(self.seed)
        emb = rng.standard_normal((1, L, self.text_dim)).astype(np.float32) * 0.01
        return mx.array(emb)


# ---------------------------------------------------------------------------
# Q3-VL truncated loader (Phase 8)
# ---------------------------------------------------------------------------
#
# Sketch (not implemented in Phase 6):
#
#   1. mlx_lm.load("mlx-community/Qwen3-VL-32B-Instruct-4bit") -> model, tok
#   2. Wrap model.forward with an early-exit at layer 50 (patch the
#      transformer_forward function or reimplement forward through blocks[:50]).
#   3. Return hidden state pre-norm, cast to fp32.
#
# Requires the vision block to be *optional* — we only need text conditioning
# for the smoke test (no image tokens in the prompt).
