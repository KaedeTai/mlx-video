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

# ---------------------------------------------------------------------------
# Phase 8.9-c: vision-token splicing constants + helpers
# ---------------------------------------------------------------------------

VISION_START = 151652    # <|vision_start|>
VISION_END = 151653      # <|vision_end|>
IMAGE_PAD = 151655       # <|image_pad|>  (positions to be replaced by vision tokens)
VIDEO_PAD = 151656       # <|video_pad|>


def _process_h3_image(
    image_hwc: "np.ndarray",
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    merge_size: int = 2,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
):
    """Preprocess an HWC [0,1] RGB image for the H3 Qwen3-VL vision tower.

    Port of ComfyUI ``comfy/text_encoders/qwen_vl.py::process_qwen2vl_images``
    with H3-specific params (patch=16, mean/std=0.5 -> 2x - 1).

    Returns
    -------
    flatten_patches : np.ndarray[float32]
        Shape ``[grid_h * grid_w, 3 * temporal_patch * patch * patch]`` — the
        pixel-value input for ``vision_tower.patch_embed``.
    grid_thw_np : np.ndarray[int64]
        Shape ``[1, 3]`` — the ``(grid_t=1, grid_h, grid_w)`` header.
    """
    import math
    from PIL import Image as _Image
    H, W, C = image_hwc.shape
    assert C == 3, f"expected RGB image, got {C} channels"

    factor = patch_size * merge_size  # 32
    h_bar = round(H / factor) * factor
    w_bar = round(W / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((H * W) / max_pixels)
        h_bar = max(factor, math.floor(H / beta / factor) * factor)
        w_bar = max(factor, math.floor(W / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (H * W))
        h_bar = math.ceil(H * beta / factor) * factor
        w_bar = math.ceil(W * beta / factor) * factor

    img_u8 = (image_hwc.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    img_pil = _Image.fromarray(img_u8).resize((w_bar, h_bar), _Image.BILINEAR)
    img = np.asarray(img_pil, dtype=np.float32) / 255.0  # HWC in [0,1]

    # Normalize: mean=std=0.5 -> 2x - 1
    img = img * 2.0 - 1.0
    img = img.transpose(2, 0, 1)  # CHW
    # Temporal patch = 2: duplicate frame
    img_t = np.broadcast_to(img[None, ...], (temporal_patch_size, 3, h_bar, w_bar)).copy()

    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size
    grid_t = 1

    patches = img_t.reshape(
        grid_t, temporal_patch_size, 3,
        grid_h // merge_size, merge_size, patch_size,
        grid_w // merge_size, merge_size, patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        3 * temporal_patch_size * patch_size * patch_size,
    ).astype(np.float32)
    grid_thw_np = np.array([[grid_t, grid_h, grid_w]], dtype=np.int64)
    return flatten_patches, grid_thw_np


def _add_deepstack_features(
    h: "mx.array", visual_pos_masks: "mx.array", delta: "mx.array"
) -> "mx.array":
    """Add per-visual-position features from ``delta`` into ``h``.

    ``h``               : [B=1, L, hidden]
    ``visual_pos_masks``: [B=1, L] bool
    ``delta``           : [n_vision, hidden]  (n_vision == mask.sum())
    """
    B = h.shape[0]
    updated = []
    for b in range(B):
        mask_np = np.asarray(visual_pos_masks[b])
        idxs_np = np.where(mask_np)[0]
        if idxs_np.shape[0] == 0:
            updated.append(h[b])
            continue
        idxs = mx.array(idxs_np, dtype=mx.uint32)
        row = mx.array(h[b])
        row = row.at[idxs].add(delta.astype(row.dtype))
        updated.append(row)
    return mx.stack(updated, axis=0)


def _minimax_token_tags(input_ids_1d: "np.ndarray") -> "mx.array":
    """Compute per-position modality tags: 0 for vision-block positions
    (including flanking ``<|vision_start|>`` / ``<|vision_end|>``), 1 for text.

    Matches ComfyUI ``comfy/text_encoders/minimax.py::token_tags_from_embeds_info``
    but works from raw token ids (widening the vision span by one on each side
    is unnecessary here — we explicitly include both markers in the span).
    """
    tags = np.ones(input_ids_1d.shape[0], dtype=np.int32)
    starts = np.where(input_ids_1d == VISION_START)[0]
    ends = np.where(input_ids_1d == VISION_END)[0]
    for s in starts:
        matching = ends[ends > s]
        if len(matching):
            e = int(matching[0])
            tags[s:e + 1] = 0
    return mx.array(tags)



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
               has_ref_audio: bool = False) -> mx.array:
        L = min(max(4, len(prompt.split()) * 2), self.max_len)
        rng = np.random.default_rng(self.seed)
        emb = rng.standard_normal((1, L, self.text_dim)).astype(np.float32) * 0.01
        return mx.array(emb)


# ---------------------------------------------------------------------------
# Ref2VA prompt prefixing (matches ComfyUI comfy/text_encoders/minimax.py)
# ---------------------------------------------------------------------------


def format_ref2va_prompt(prompt: str, has_ref_image: bool, has_ref_audio: bool) -> str:
    """Prepend "<Picture 1>: " / "<Audio 1>: " prefixes per ref2va format.

    ComfyUI splices actual vision tokens between VISION_START/VISION_END for
    the image; we ship only the text prefix in Phase 8.9-b (still restores
    intelligibility because it matches the H3 training presentation).
    """
    parts = []
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
               has_ref_audio: bool = False) -> mx.array:
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
        model_path: str = "~/mlx-video/mlx-models/H3-TextEncoder-MLX-mxfp4",
        truncate_layer: int = TRUNC_LAYERS,
        load_vision: bool = True,
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

        # Phase 8.9-d: if the saved checkpoint is quantized (mxfp4/affine),
        # nn.quantize the target model with a predicate that matches ONLY
        # the layers whose ``.scales`` are present in ``weights``. This
        # lets us keep vision tower + deepstack + embed_tokens in bf16
        # while the 50 language-model decoder blocks live at 4-bit.
        _q = config.get("quantization") or config.get("quantization_config")
        if _q is not None:
            import mlx.nn as _nn
            def _quant_pred(_p, _m):
                if not hasattr(_m, "to_quantized"):
                    return False
                return f"{_p}.scales" in weights
            _nn.quantize(
                model,
                group_size=_q["group_size"], bits=_q["bits"],
                mode=_q.get("mode", "affine"),
                class_predicate=_quant_pred,
            )

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
        elif self.load_vision and not hasattr(model, "vision_tower"):
            raise RuntimeError(
                "H3TextEncoderBridge(load_vision=True): loaded model has "
                "no `vision_tower` attribute (checkpoint mismatch?)"
            )

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

    def _encode_with_vision(
        self,
        prompt: str,
        ref_image_hwc: "np.ndarray",
        has_ref_audio: bool,
    ) -> tuple:
        """Phase 8.9-c full vision path: splice vision tokens into text stream.

        Steps (mirroring ComfyUI ``comfy/text_encoders/minimax.py`` + Qwen3-VL):

        1. Preprocess ``ref_image_hwc`` -> ``(flatten_patches, grid_thw)``.
        2. Build token stream ``"<Picture 1>: " <vision_start> <image_pad>*N
           <vision_end> ["<Audio 1>: "] <prompt>``.
        3. Run vision tower to get ``(merged, deepstack_list)``.
        4. Splice merged features into ``embed_tokens`` output at image_pad
           positions.
        5. Compute MRoPE ``position_ids`` from ``get_rope_index``.
        6. Forward through 50 LM layers, adding deepstack features at the
           first ``len(deepstack_list)`` layers on visual positions.
        7. NO final norm (H3 spec: unnormalized after layer 50).

        Returns
        -------
        (hidden, token_tags) : ([1, L, 5120] mx.array, [L] int mx.array)
        """
        flatten, grid_thw_np = _process_h3_image(ref_image_hwc)
        grid_h = int(grid_thw_np[0, 1])
        grid_w = int(grid_thw_np[0, 2])
        n_vision = (grid_h * grid_w) // 4  # spatial_merge_size**2

        text_ids_pic = self._tokenize("<Picture 1>: ")
        text_ids_prompt = self._tokenize(prompt) if prompt else []
        text_ids_audio = self._tokenize("<Audio 1>: ") if has_ref_audio else []
        input_ids_list = (
            text_ids_pic
            + [VISION_START] + [IMAGE_PAD] * n_vision + [VISION_END]
            + text_ids_audio
            + text_ids_prompt
        )
        if not input_ids_list:
            input_ids_list = [151643]
        input_ids = mx.array([input_ids_list], dtype=mx.int32)  # [1, L]

        # ---- Vision tower ----
        vision_tower = self._model.vision_tower
        pixel_dtype = vision_tower.patch_embed.proj.weight.dtype
        pixel_values = mx.array(flatten).astype(pixel_dtype)
        grid_thw = mx.array(grid_thw_np)
        vision_hidden, deepstack_lists = vision_tower(pixel_values, grid_thw)
        # vision_hidden: [n_vision, 5120]; deepstack_lists: list of 3 arrays [n_vision, 5120]

        # ---- Splice vision features into input embeddings ----
        from mlx_vlm.models.qwen3_vl.qwen3_vl import Model as _Q3VLModel
        inputs_embeds = self._lang.model.embed_tokens(input_ids)
        inputs_embeds, image_mask = _Q3VLModel.merge_input_ids_with_image_features(
            vision_hidden, inputs_embeds, input_ids,
            image_token_index=IMAGE_PAD, video_token_index=VIDEO_PAD,
        )
        visual_pos_masks = image_mask[..., 0]  # [1, L] bool

        # ---- MRoPE position ids ----
        position_ids, _ = self._lang.get_rope_index(
            input_ids, grid_thw, None, None
        )

        # ---- 50 LM layers with deepstack injection, NO final norm ----
        from mlx_vlm.models.base import create_attention_mask
        h = inputs_embeds
        mask = create_attention_mask(h, [None] * len(self._lang.model.layers))
        n_deep = len(deepstack_lists) if deepstack_lists else 0
        for layer_idx, layer in enumerate(self._lang.model.layers):
            h = layer(h, mask=mask, cache=None, position_ids=position_ids)
            if n_deep > 0 and layer_idx < n_deep:
                h = _add_deepstack_features(
                    h, visual_pos_masks, deepstack_lists[layer_idx]
                )

        if self.dtype_fp32_out:
            h = h.astype(mx.float32)
        mx.eval(h)

        tags = _minimax_token_tags(np.asarray(input_ids[0]))
        return h, tags

    def encode(
        self,
        prompt: str,
        has_ref_image: bool = False,
        has_ref_audio: bool = False,
        ref_image: Optional["np.ndarray"] = None,
        return_token_tags: bool = False,
    ):
        """Return the layer-50 unnormalized hidden state as ``[1, L, 5120]``.

        Two paths:

        - **Phase 8.9-c (vision-aware, default when ``ref_image`` is given)**:
          Preprocesses ``ref_image`` (HWC float32 in ``[0, 1]``), runs the
          Qwen3-VL vision tower, splices vision tokens between
          ``<|vision_start|>`` and ``<|vision_end|>`` per the H3/ComfyUI
          ref2va format, and injects deepstack features into the first three
          LM layers at visual positions.

        - **Phase 8.9-b fallback (text-only)**: prepends the
          ``"<Picture 1>: "`` / ``"<Audio 1>: "`` text prefixes but does not
          splice vision tokens. Used when ``ref_image`` is None.

        Parameters
        ----------
        prompt : str
            Free-form prompt text (raw, no chat template).
        has_ref_image : bool
            When True and no ``ref_image`` array is provided, uses the
            text-only ``"<Picture 1>: "`` prefix.
        has_ref_audio : bool
            Prepends ``"<Audio 1>: "`` (audio tokens never enter Qwen — the
            audio branch conditions on the audio VAE latent instead).
        ref_image : np.ndarray, optional
            Full HWC RGB image in ``[0, 1]`` float32; when provided, engages
            the vision-token splicing path.
        return_token_tags : bool
            When True, returns ``(hidden, minimax_token_tags)`` instead of
            just ``hidden``. Vision block positions (including the flanking
            ``<|vision_start|>`` / ``<|vision_end|>``) get tag ``0``; text
            positions get tag ``1``. The DiT's adaLN reads this to route each
            text-stream position to a modality-specific shift/scale.
        """
        if ref_image is not None:
            hidden, tags = self._encode_with_vision(
                prompt, ref_image, has_ref_audio
            )
        else:
            prefixed = format_ref2va_prompt(prompt, has_ref_image, has_ref_audio)
            input_ids = self._tokenize(prefixed)
            hidden = self._forward_text_only(input_ids)
            tags = mx.ones((hidden.shape[1],), dtype=mx.int32)
        if return_token_tags:
            return hidden, tags
        return hidden
