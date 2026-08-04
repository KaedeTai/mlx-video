"""Text encoder wiring for MiniMax H3.

MiniMax H3 conditions on the **unnormalized layer-50 hidden state** of
Qwen3-VL-32B (64 layers total, so layer 50 is a mid-network cut). See
``comfy/text_encoders/minimax.py`` in the reference.

Phase 8-1: replaces the Phase-7 ``DummyTextEncoder`` with a real
``TextEncoderBridge`` that wraps ``mlx-community/Qwen3-VL-32B-Instruct-4bit``
(loaded via mlx-vlm), truncates the decoder stack at layer 50, and returns
the raw pre-norm hidden state cast to fp32.

The bridge supports **text-only prompts** in this cut. Vision-block
conditioning through Qwen3-VL (reference images / video segments spliced into
the token stream per the ComfyUI ref) is out of scope here -- reference images
still flow through the video VAE + DiT ref-block path. See Phase 9 notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import numpy as np


TRUNC_LAYERS = 50  # H3 uses Qwen3-VL layers [0, TRUNC_LAYERS)


@dataclass
class DummyTextEncoder:
    """Placeholder text encoder that returns zero-init embeddings.

    Kept for smoke tests / CI where loading the 18 GB Q4 checkpoint is too
    heavy. Video quality is uncontrolled.
    """
    text_dim: int = 5120
    max_len: int = 128
    seed: int = 0

    def encode(self, prompt: str) -> mx.array:
        L = min(max(4, len(prompt.split()) * 2), self.max_len)
        rng = np.random.default_rng(self.seed)
        emb = rng.standard_normal((1, L, self.text_dim)).astype(np.float32) * 0.01
        return mx.array(emb)


# ---------------------------------------------------------------------------
# Real Qwen3-VL-32B truncated encoder (Phase 8-1)
# ---------------------------------------------------------------------------


class TextEncoderBridge:
    """Wraps mlx-community/Qwen3-VL-32B-Instruct-4bit for H3 conditioning.

    Only text-side layers are used (embed_tokens + first ``truncate_layer``
    decoder blocks). The vision tower is NOT loaded when
    ``load_vision=False``, keeping RAM ~18 GB instead of ~19 GB.

    Parameters
    ----------
    model_path : str | Path
        Local MLX checkpoint directory.
    truncate_layer : int
        Number of decoder blocks to run (inclusive-exclusive), default 50.
    load_vision : bool
        If False (default), skip the vision tower -- vision-block prompt
        splicing is not exposed at this layer of the port.
    dtype_fp32_out : bool
        If True (default), cast the returned hidden state to fp32 to match
        the DiT ingestion path.
    """

    def __init__(
        self,
        model_path: str,
        truncate_layer: int = TRUNC_LAYERS,
        load_vision: bool = False,
        dtype_fp32_out: bool = True,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser())
        self.truncate_layer = int(truncate_layer)
        self.load_vision = load_vision
        self.dtype_fp32_out = dtype_fp32_out

        from mlx_vlm import load as _mlx_vlm_load

        model, processor = _mlx_vlm_load(self.model_path, lazy=False)
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)

        lang = model.language_model
        lang.model.layers = lang.model.layers[: self.truncate_layer]

        if not self.load_vision and hasattr(model, "vision_tower"):
            del model.vision_tower

        self._model = model
        self._lang = lang
        self.text_dim = lang.args.hidden_size  # 5120

    def _encode_no_vision(self, prompt: str) -> mx.array:
        input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(input_ids) == 0:
            input_ids = [151643]  # pad token, matches Comfy fallback

        ids = mx.array([input_ids], dtype=mx.int32)  # [1, L]
        h = self._lang.model.embed_tokens(ids)  # [1, L, 5120]

        B, L, _ = h.shape

        pos = mx.arange(L, dtype=mx.int32)
        pos = mx.broadcast_to(pos[None, :], (B, L))
        pos = mx.broadcast_to(pos[None, ...], (3, B, L))

        from mlx_vlm.models.base import create_attention_mask
        mask = create_attention_mask(h, [None] * len(self._lang.model.layers))

        for layer in self._lang.model.layers:
            h = layer(h, mask=mask, cache=None, position_ids=pos)

        if self.dtype_fp32_out:
            h = h.astype(mx.float32)
        mx.eval(h)
        return h

    def encode(self, prompt: str) -> mx.array:
        return self._encode_no_vision(prompt)
