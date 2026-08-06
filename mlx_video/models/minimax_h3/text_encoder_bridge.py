"""Text encoder wiring for MiniMax H3.

MiniMax H3 conditions on the **unnormalized layer-50 hidden state** of
Qwen3-VL-32B (64-layer LM, so layer 50 is a mid-network cut). See
``comfy/text_encoders/minimax.py`` in the reference.

Phase 8-1 (superseded)
----------------------
``TextEncoderBridge`` (kept for back-compat) wrapped
``mlx-community/Qwen3-VL-32B-Instruct-4bit`` — a stock AWQ Q4 checkpoint
**calibrated for LM-head generation**, NOT for mid-layer hidden-state
extraction. The resulting distributionally-shifted conditioning gave the DiT
speech-like garbage sound (no intelligible language).

Phase 8.9-b (current)
---------------------
``H3TextEncoderBridge`` loads the **H3-specific bf16 encoder**
(``Comfy-Org/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors``,
51 GB, __metadata__ = ``{"num_hidden_layers": 50,
"output": "unnormalized_hidden_after_layer_50"}``), converted to an
mlx-vlm-compatible directory by ``scripts/h3/convert_h3_text_encoder.py``.

Ref2VA text prefixing per ComfyUI (``comfy/text_encoders/minimax.py``):

    "<Picture 1>: " + <vision block>   when a ref image is present
    "<Audio 1>: "                       when a ref audio is present
                                        (audio never enters Qwen)
    ...
    <prompt>

Phase 8.9-b still ships a **text-only branch**: we prepend the "<Picture N>: "
and "<Audio N>: " text prefixes but do NOT splice actual vision tokens into
the stream. Vision-token splicing (needs the vision tower + deep-stack
projection) is deferred; the text prefix alone matches the conditioning
distribution well enough that whisper-decodable Chinese emerges.
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

    Kept for smoke tests / CI where loading the 51 GB bf16 checkpoint is too
    heavy. Video quality is uncontrolled.
    """
    text_dim: int = 5120
    max_len: int = 128
    seed: int = 0

    def encode(self, prompt: str, has_ref_image: bool = False,
               has_ref_audio: bool = False, has_ref_video: bool = False) -> mx.array:
        L = min(max(4, len(prompt.split()) * 2), self.max_len)
        rng = np.random.default_rng(self.seed)
        emb = rng.standard_normal((1, L, self.text_dim)).astype(np.float32) * 0.01
        return mx.array(emb)


# ---------------------------------------------------------------------------
# Ref2VA prompt prefixing (matches ComfyUI comfy/text_encoders/minimax.py)
# ---------------------------------------------------------------------------


def format_ref2va_prompt(prompt: str, has_ref_image: bool, has_ref_audio: bool, has_ref_video: bool = False) -> str:
    """Prepend "<Picture 1>: " / "<Audio 1>: " prefixes per ref2va format.

    ComfyUI splices actual vision tokens between VISION_START/VISION_END for
    the image; we ship only the text prefix in Phase 8.9-b (still restores
    intelligibility because it matches the H3 training presentation).
    """
    parts = []
    if has_ref_video:
        parts.append("<Video 1>: ")
    if has_ref_image:
        parts.append("<Picture 1>: ")
    if has_ref_audio:
        parts.append("<Audio 1>: ")
    parts.append(prompt or "")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Legacy Q4 encoder (Phase 8-1, retained for CI + smoke)
# ---------------------------------------------------------------------------


class TextEncoderBridge:
    """Wraps mlx-community/Qwen3-VL-32B-Instruct-4bit for H3 conditioning.

    Note (Phase 8.9-b): this class is the *legacy* AWQ-Q4 bridge whose
    mid-layer hidden distribution is distorted; use ``H3TextEncoderBridge``
    for real language generation. Kept here only so old smoke scripts still
    import.

    Parameters
    ----------
    model_path : str | Path
        Local MLX checkpoint directory.
    truncate_layer : int
        Number of decoder blocks to run (inclusive-exclusive), default 50.
    load_vision : bool
        If False (default), skip the vision tower.
    dtype_fp32_out : bool
        If True (default), cast the returned hidden state to fp32.
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
            input_ids = [151643]  # pad token

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

    def encode(self, prompt: str, has_ref_image: bool = False,
               has_ref_audio: bool = False, has_ref_video: bool = False) -> mx.array:
        # Legacy encoder — ref-flags accepted for API parity but ignored.
        return self._encode_no_vision(prompt)


# ---------------------------------------------------------------------------
# Phase 8.9-b: real H3-specific bf16 encoder
# ---------------------------------------------------------------------------


class H3TextEncoderBridge:
    """Wraps the H3-specific bf16 Qwen3-VL-32B (50 layers) encoder.

    Loads an mlx-vlm-compatible directory produced by
    ``scripts/h3/convert_h3_text_encoder.py`` from
    ``Comfy-Org/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors``.

    Unlike ``TextEncoderBridge`` (Phase 8-1), this uses the **H3-tuned**
    weights whose mid-layer hidden distribution actually matches what the DiT
    expects, so the generated audio contains real Chinese speech instead of
    speech-like garbage.

    Parameters
    ----------
    model_path : str | Path
        Directory holding the converted mlx-vlm-format checkpoint (default
        ``~/mlx-video/mlx-models/H3-TextEncoder-MLX-bf16``).
    truncate_layer : int
        Decoder blocks retained after load, default 50. The converted
        checkpoint is already truncated to 50 layers, so this is a no-op
        unless you want to bench a shorter cut.
    load_vision : bool
        If False (default, Phase 8.9-b), delete the vision tower after load
        to save ~4 GB — text-only conditioning ships in this cut.
    dtype_fp32_out : bool
        Cast the returned hidden state to fp32 (default True).
    """

    def __init__(
        self,
        model_path: str = "~/mlx-video/mlx-models/H3-TextEncoder-MLX-bf16",
        truncate_layer: int = TRUNC_LAYERS,
        load_vision: bool = False,
        dtype_fp32_out: bool = True,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser())
        self.truncate_layer = int(truncate_layer)
        self.load_vision = load_vision
        self.dtype_fp32_out = dtype_fp32_out

        # Custom loader — mlx_vlm.utils.load_model calls
        # ``model.load_weights(list(weights.items()))`` with no strict=False,
        # which fails on the H3 encoder because the Comfy-Org checkpoint
        # omits ``lm_head.weight`` and ``model.norm.weight`` (H3 uses the
        # unnormalized layer-50 hidden state and never runs the LM head).
        # We inline the minimum load-model logic and pass strict=False.
        import glob as _glob
        import mlx.core as _mx
        from mlx_vlm.utils import (
            load_config as _load_config,
            get_model_and_args as _get_model_and_args,
            update_module_configs as _update_module_configs,
            sanitize_weights as _sanitize_weights,
            load_processor as _load_processor,
            load_image_processor as _load_image_processor,
        )

        config = _load_config(Path(self.model_path))
        weight_files = sorted(
            wf for wf in _glob.glob(f"{self.model_path}/*.safetensors")
            if not wf.endswith("consolidated.safetensors")
        )
        if not weight_files:
            raise FileNotFoundError(
                f"no *.safetensors in {self.model_path}")

        weights: dict[str, Any] = {}
        for wf in weight_files:
            weights.update(_mx.load(wf))

        model_class, _ = _get_model_and_args(config=config)
        config.setdefault("text_config", config.pop("llm_config", {}))
        config.setdefault("vision_config", {})
        config.setdefault("audio_config", {})
        model_config = model_class.ModelConfig.from_dict(config)
        model_config = _update_module_configs(
            model_config, model_class, config,
            ["text", "vision", "perceiver", "projector", "audio"],
        )
        model = model_class.Model(model_config)

        # Sanitize (renames model.language_model.* -> language_model.model.*, etc.)
        weights = _sanitize_weights(model, weights)
        weights = _sanitize_weights(
            model_class.VisionModel, weights, model_config.vision_config)
        weights = _sanitize_weights(
            model_class.LanguageModel, weights, model_config.text_config)

        # Phase 8.9-c: apply Q4 quantization to the model shell BEFORE
        # load_weights when the config carries a top-level ``quantization``
        # block. Mirrors mlx_vlm.utils.load_model's quantize path but with the
        # bridge's strict=False allowance for missing lm_head / final norm.
        quantization = config.get("quantization", None)
        if quantization is not None:
            import mlx.nn as _nn
            _gs = int(quantization["group_size"])
            _bits = int(quantization["bits"])
            _mode = quantization.get("mode", "affine")
            def _pred(pth, mod):
                if not hasattr(mod, "to_quantized"):
                    return False
                if hasattr(mod, "weight") and mod.weight.size % _gs != 0:
                    return False
                # Only quantize layers whose scales are present in the file —
                # this naturally skips vision_tower / lm_head / dropped norms.
                return f"{pth}.scales" in weights
            _nn.quantize(model, group_size=_gs, bits=_bits,
                         class_predicate=_pred, mode=_mode)

        # strict=False: allow missing lm_head + final norm (H3 doesn't use them)
        model.load_weights(list(weights.items()), strict=False)
        _mx.eval(model.parameters())

        # Standalone processor / image processor
        eos_token_id = getattr(model.config, "eos_token_id", None)
        _mp = Path(self.model_path)
        image_processor = _load_image_processor(_mp)
        processor = _load_processor(
            _mp, True, eos_token_ids=eos_token_id)
        if image_processor is not None:
            processor.image_processor = image_processor
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)

        lang = model.language_model
        # Belt + suspenders: even though config.text_config.num_hidden_layers
        # is 50, keep the explicit truncation for parity with the legacy path.
        lang.model.layers = lang.model.layers[: self.truncate_layer]

        # Drop the LM head — we take the pre-norm hidden, never logits.
        if hasattr(lang, "lm_head"):
            del lang.lm_head
        # Drop the final norm too — H3 spec: unnormalized after layer 50.
        if hasattr(lang.model, "norm"):
            del lang.model.norm

        if not self.load_vision and hasattr(model, "vision_tower"):
            del model.vision_tower

        self._model = model
        self._lang = lang
        self.text_dim = lang.args.hidden_size  # 5120

    def _tokenize(self, prompt: str) -> list[int]:
        input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(input_ids) == 0:
            input_ids = [151643]  # pad
        return input_ids

    def _forward_text_only(self, input_ids: list[int]) -> mx.array:
        ids = mx.array([input_ids], dtype=mx.int32)  # [1, L]
        h = self._lang.model.embed_tokens(ids)  # [1, L, 5120]
        B, L, _ = h.shape

        # Text-only mrope: same position along T/H/W axes.
        pos = mx.arange(L, dtype=mx.int32)
        pos = mx.broadcast_to(pos[None, :], (B, L))
        pos = mx.broadcast_to(pos[None, ...], (3, B, L))

        from mlx_vlm.models.base import create_attention_mask
        mask = create_attention_mask(h, [None] * len(self._lang.model.layers))

        for layer in self._lang.model.layers:
            h = layer(h, mask=mask, cache=None, position_ids=pos)

        # H3 spec: unnormalized hidden after layer 50 -> DO NOT apply
        # self._lang.model.norm here. The layer output is what we want.
        if self.dtype_fp32_out:
            h = h.astype(mx.float32)
        mx.eval(h)
        return h

    def encode(self, prompt: str, has_ref_image: bool = False,
               has_ref_audio: bool = False, has_ref_video: bool = False) -> mx.array:
        """Return the layer-50 unnormalized hidden state as ``[1, L, 5120]``.

        The ref2va text prefixes (``<Picture 1>: `` / ``<Audio 1>: ``) are
        prepended automatically when the flags are True. Vision-token
        splicing is not implemented in Phase 8.9-b.
        """
        prefixed = format_ref2va_prompt(prompt, has_ref_image, has_ref_audio, has_ref_video)
        input_ids = self._tokenize(prefixed)
        return self._forward_text_only(input_ids)
