"""MiniMax H3 top-level DiT — MLX port.

Port of ``MiniMaxH3Model._forward`` in ``comfy/ldm/minimax/model.py`` lines 380-646.

The ``__call__`` signature mirrors ComfyUI's ``_forward``:
  ``__call__((video, audio), timestep, context, payload=None) ->
      [neg_video_velocity, neg_slope_a_scaled_audio_velocity]``

The sampler drives a single flat ODE ``dX/dsigma_v = (X - denoised)/sigma_v``.
Scaling the audio branch's velocity by ``time_shift_slope(sigma_v, shift_v,
shift_a)`` makes that same ODE equal to the audio stream's true ODE on its own
shifted schedule.

Key MLX adaptations
-------------------
- Batch size is always 1 (matches reference constraint).
- The packed stream is [S, hidden]; each block is a plain functional pass.
- Position IDs and unique-timestep list are computed in numpy (fast, static);
  RoPE table and time embedding cache once per forward.
- The "assemble h[a:b] from embed rows" step uses list-append + concatenate
  (no item assignment).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .attention import H3Attention  # noqa: F401 (re-export)
from .blocks import (
    AdalnProj,
    DiTBlock,
    FinalLayer,
    H3MLP,  # noqa: F401
    RefinerBlock,  # noqa: F401
    TimeEmbedder,
    TokenRefiner,
    _mod_gate,  # noqa: F401
    _mod_scale_shift,  # noqa: F401
)
from .config import MiniMaxH3Config
from .packed_layout import (
    AUDIO_COND_TIMESTEP,
    VISUAL_COND_TIMESTEP,
    PackedLayout,
)
from .rope import build_rope_table


# ---------------------------------------------------------------------------
# Utility: time_shift_sigma / time_shift_slope (dual-schedule map)
# ---------------------------------------------------------------------------


def time_shift_sigma(sigma: float, from_shift: float, to_shift: float) -> float:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def time_shift_slope(sigma: float, from_shift: float, to_shift: float) -> float:
    """d(sigma_to)/d(sigma_from) at the same base-grid point."""
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return (to_shift * (1.0 + (from_shift - 1.0) * base) ** 2) / (
        from_shift * (1.0 + (to_shift - 1.0) * base) ** 2
    )


# ---------------------------------------------------------------------------
# Patchify / pack helpers (MLX channel-first for latents, matching reference IO)
# ---------------------------------------------------------------------------


def patchify_video(latent: mx.array, patch_size: Tuple[int, int, int] = (1, 2, 2)) -> mx.array:
    """[B=1, C, T, H, W] -> [B*t*h*w, C*pt*ph*pw]."""
    b, c, t_full, h_full, w_full = latent.shape
    pt, ph, pw = patch_size
    t, h, w = t_full // pt, h_full // ph, w_full // pw
    x = latent.reshape(b, c, t, pt, h, ph, w, pw)
    # nctrhpwq -> nthwcrpq
    x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(b * t * h * w, c * pt * ph * pw)


def unpatchify_video(rows: mx.array, t: int, h: int, w: int,
                     c: int = 24, patch_size: Tuple[int, int, int] = (1, 2, 2)) -> mx.array:
    pt, ph, pw = patch_size
    x = rows.reshape(-1, t, h, w, c, pt, ph, pw)
    # nthwcrpq -> nctrhpwq
    x = x.transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(-1, c, t * pt, h * ph, w * pw)


def pack_audio(latent: mx.array) -> mx.array:
    """[B=1, C=32, ch=2, T] -> [ch*T, 32] channel-major."""
    b, c, ch, t = latent.shape
    # latent[0]: [C, ch, T] -> permute(1, 2, 0) -> [ch, T, C] -> reshape [ch*T, C]
    x = latent[0].transpose(1, 2, 0)
    return x.reshape(ch * t, c)


def unpack_audio(rows: mx.array, ch: int = 2) -> mx.array:
    """[ch*T, C] -> [1, C, ch, T]."""
    t = rows.shape[0] // ch
    C = rows.shape[-1]
    x = rows.reshape(ch, t, C)
    # permute (2, 0, 1) then unsqueeze batch
    x = x.transpose(2, 0, 1)
    return x[None, ...]


def _pad_to_patch_size(x: mx.array, patch_size: Tuple[int, int, int]) -> mx.array:
    """Right/bottom pad a [B, C, T, H, W] tensor with zeros so each spatial dim is patch-aligned."""
    b, c, t, h, w = x.shape
    pt, ph, pw = patch_size
    t_pad = (pt - t % pt) % pt
    h_pad = (ph - h % ph) % ph
    w_pad = (pw - w % pw) % pw
    if t_pad or h_pad or w_pad:
        pads = [(0, 0), (0, 0), (0, t_pad), (0, h_pad), (0, w_pad)]
        x = mx.pad(x, pads)
    return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class RopeModule(nn.Module):
    """Container for the inv_freq buffer (keeps the weight name ``rope.inv_freq``)."""
    def __init__(self, freq_len: int):
        super().__init__()
        self.inv_freq = mx.zeros((freq_len,), dtype=mx.float32)


class MiniMaxH3Model(nn.Module):
    """MiniMax H3 audio-video DiT."""

    def __init__(self, config: MiniMaxH3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.patch_size = tuple(config.patch_size)
        self.latents_dim = config.latents_dim
        self.audio_latents_dim = config.audio_latents_dim
        self.sigma_shift_video = config.sigma_shift_video
        self.sigma_shift_audio = config.sigma_shift_audio
        self.use_adaln_curves = config.adaln_curve_grid is not None
        self.apply_silu = not self.use_adaln_curves

        video_patch_dim = config.video_patch_dim

        # fp32 patch projections (matches checkpoint dtype)
        self.video_patch_proj = nn.Linear(video_patch_dim, config.hidden_size, bias=True)
        self.audio_patch_proj = nn.Linear(config.audio_latents_dim, config.hidden_size, bias=True)
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size, bias=True)

        if self.use_adaln_curves:
            self.adaln_t_table = mx.zeros((config.adaln_curve_grid, config.time_embed_dim), dtype=mx.float32)
        else:
            self.time_embedder = TimeEmbedder(config.timestep_input_dim,
                                              config.time_embed_hidden_size,
                                              config.time_embed_dim)

        self.rope = RopeModule(config.rope_inv_freq_len)

        self.token_refiner = TokenRefiner(
            config.token_refiner_num_layers, config.hidden_size,
            config.num_attention_heads, config.attention_head_dim,
            config.ffn_hidden_size, config.norm_eps, config.qk_norm_eps,
            config.final_norm_eps,
        )
        self.blocks = [
            DiTBlock(config.hidden_size, config.num_attention_heads, config.attention_head_dim,
                     config.ffn_hidden_size, config.time_embed_dim,
                     config.norm_eps, config.qk_norm_eps, apply_silu=self.apply_silu)
            for _ in range(config.num_layers)
        ]
        self.final_layer = FinalLayer(
            config.hidden_size, config.time_embed_dim,
            video_patch_dim, config.audio_latents_dim,
            config.final_norm_eps, apply_silu=self.apply_silu,
        )

    # ------------------------------------------------------------------
    # Cond stream helpers
    # ------------------------------------------------------------------

    def _cond_video_rows(self, payload: Dict[str, Any]) -> Optional[mx.array]:
        rows = []
        aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
        seed = int(payload.get("seed", 0))
        for z in payload.get("cond_video_latents", []):
            r = patchify_video(z.astype(mx.float32), self.patch_size)
            if aug < 1.0:
                rng = np.random.default_rng(seed)
                noise = mx.array(rng.standard_normal(size=r.shape).astype(np.float32))
                r = aug * r + (1.0 - aug) * noise
            rows.append(r)
        return mx.concatenate(rows, axis=0) if rows else None

    def _cond_audio_rows(self, payload: Dict[str, Any]) -> Optional[mx.array]:
        rows = []
        aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
        seed = int(payload.get("seed", 0)) + 1
        for z in payload.get("cond_audio_latents", []):
            r = pack_audio(z.astype(mx.float32))
            if aug < 1.0:
                rng = np.random.default_rng(seed)
                noise = mx.array(rng.standard_normal(size=r.shape).astype(np.float32))
                r = aug * r + (1.0 - aug) * noise
            rows.append(r)
        return mx.concatenate(rows, axis=0) if rows else None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def preprocess_text_embeds(self, text_states: mx.array) -> mx.array:
        """[B, L, text_dim] Qwen states -> [B, L, hidden] refined text embeds."""
        if text_states.shape[-1] == self.hidden_size:
            return text_states
        return self.token_refiner(self.condition_proj(text_states[0]))[None, ...]

    def __call__(
        self,
        x: Sequence[mx.array],
        timestep: mx.array,
        context: mx.array,
        payload: Optional[Dict[str, Any]] = None,
    ) -> List[mx.array]:
        video_x, audio_x = x[0], x[1]
        orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        video_x = _pad_to_patch_size(video_x, self.patch_size)
        if video_x.shape[0] != 1:
            raise ValueError("MiniMax H3 supports batch size 1")
        payload = payload or {}
        # Compute dtype driven by context (mirrors reference's `dtype = context.dtype`)
        compute_dtype = context.dtype

        latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
        audio_t = audio_x.shape[-1]
        text_len = context.shape[1]

        # Layout can be cached across steps in the payload
        layout = payload.get("layout")
        if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
            layout = PackedLayout(
                text_len, latent_t, lat_h, lat_w, audio_t,
                keyframes=payload.get("keyframes"),
                refs=payload.get("refs"),
                frame_count=payload.get("frame_count"),
            )

        # ---- Timestep math ----
        shift_v = float(self.sigma_shift_video)
        shift_a = float(self.sigma_shift_audio)
        sigma_v_val = float(mx.maximum(timestep.flatten()[0] / 1000.0, mx.array(1e-6)).item())
        t_v = float(1.0 - sigma_v_val)
        t_a = float(1.0 - time_shift_sigma(sigma_v_val, shift_v, shift_a))

        vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
        aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
        has_vis_cond = any(k in ("cond", "ref_img") for _, _, k in layout.segments)
        has_aud_cond = any(k == "ref_audio" for _, _, k in layout.segments)
        seg_t = {
            "text": t_v, "video": t_v, "audio": t_a,
            "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
            # ref-audio rows are pinned at condition_audio_timestep (== aud_aug),
            # NOT max(t_a, aud_aug). Pipenetwork uses aud_aug=0.0 (fully clean ref
            # content); the prior max(...) held ref rows at t=1.0 (pure noise) when
            # aud_aug was mistakenly 1.0 (see AUDIO_COND_TIMESTEP fix in
            # packed_layout.py). This is the second half of the ref_blocks port.
            "ref_audio": aud_aug,
        }
        distinct = {t_v, t_a}
        if has_vis_cond:
            distinct.add(seg_t["cond"])
        if has_aud_cond:
            distinct.add(seg_t["ref_audio"])
        unique_t = sorted(distinct)
        t_row = {t: i for i, t in enumerate(unique_t)}
        seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

        text_tags = payload.get("text_token_tags")
        mod_segments: List[Tuple[int, int, int]] = []
        for a, b, kind in layout.segments:
            row_base = t_row[seg_t[kind]] * 3
            if kind == "text" and text_tags is not None:
                tags = np.asarray(text_tags).reshape(-1).tolist()
                run_start = 0
                for i in range(1, b - a + 1):
                    if i == b - a or tags[i] != tags[run_start]:
                        mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                        run_start = i
            else:
                mod_segments.append((a, b, row_base + seg_tag[kind]))

        # ---- Row assembly (video / audio streams, then embed & scatter into h) ----
        img_update = layout.img_update
        audio_update = layout.audio_update

        video_rows = patchify_video(video_x.astype(mx.float32), self.patch_size)
        audio_rows = pack_audio(audio_x.astype(mx.float32))
        cond_video_rows = self._cond_video_rows(payload)
        cond_audio_rows = self._cond_audio_rows(payload)

        # Interleave cond/target rows respecting img_update / audio_update masks
        if cond_video_rows is not None:
            # walk the mask, producing an [N_img_total, C] tensor
            all_video_rows = _weave_by_mask(cond_video_rows, video_rows, img_update)
        else:
            all_video_rows = video_rows
        if cond_audio_rows is not None:
            all_audio_rows = _weave_by_mask(cond_audio_rows, audio_rows, audio_update)
        else:
            all_audio_rows = audio_rows

        video_embed = self.video_patch_proj(all_video_rows).astype(compute_dtype)
        audio_embed = self.audio_patch_proj(all_audio_rows).astype(compute_dtype)

        text_states = context[0]
        if text_states.shape[-1] != self.hidden_size:
            text_states = self.token_refiner(self.condition_proj(text_states))

        # Assemble h by concatenating slices in segment order (avoids item assignment)
        parts: List[mx.array] = []
        voff = 0
        aoff = 0
        for a, b, kind in layout.segments:
            n = b - a
            if kind == "text":
                parts.append(text_states)
            elif kind in ("cond", "ref_img", "video"):
                parts.append(video_embed[voff:voff + n])
                voff += n
            else:  # ref_audio / audio
                parts.append(audio_embed[aoff:aoff + n])
                aoff += n
        h = mx.concatenate(parts, axis=0)

        # ---- Time embedding (per-unique-t) ----
        t_vals = mx.array(unique_t, dtype=mx.float32)
        if self.use_adaln_curves:
            table = self.adaln_t_table
            pos = mx.clip(t_vals, 0.0, 1.0) * (table.shape[0] - 1)
            i0 = mx.clip(mx.floor(pos).astype(mx.int32), 0, table.shape[0] - 2)
            frac = (pos - i0.astype(mx.float32))[:, None]
            # Lerp table[i0] and table[i0+1]
            t0 = mx.take(table, i0, axis=0)
            t1 = mx.take(table, i0 + 1, axis=0)
            t_emb = t0 * (1.0 - frac) + t1 * frac
            t_emb = t_emb.astype(compute_dtype)
        else:
            t_emb = self.time_embedder(t_vals).astype(compute_dtype)

        # ---- Rope table (once per forward) ----
        inv_freq_np = np.asarray(self.rope.inv_freq).astype(np.float32)
        rope_table = build_rope_table(layout.position_ids, inv_freq_np, dtype=compute_dtype)

        # ---- 50 DiT blocks ----
        for block in self.blocks:
            h = block(h, t_emb, mod_segments, rope_table)

        # ---- Final layer: split video / audio slices ----
        video_seg = next((a, b, t_row[seg_t["video"]]) for a, b, k in layout.segments if k == "video")
        audio_seg = next((a, b, t_row[seg_t["audio"]]) for a, b, k in layout.segments if k == "audio")
        v_out, a_out = self.final_layer(h, t_emb, video_seg, audio_seg)

        # ---- Unpatchify + trim to orig ----
        video_out = unpatchify_video(v_out, latent_t, lat_h // 2, lat_w // 2,
                                     self.latents_dim, self.patch_size)
        video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
        audio_out = unpack_audio(a_out)

        slope_a = time_shift_slope(sigma_v_val, shift_v, shift_a)
        video_ret = (-video_out).astype(video_x.dtype)
        audio_ret = ((-slope_a) * audio_out).astype(audio_x.dtype)
        return [video_ret, audio_ret]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weave_by_mask(cond_rows: mx.array, target_rows: mx.array, update_mask: np.ndarray) -> mx.array:
    """Weave two row streams according to a boolean mask.

    ``update_mask[i] == True`` means row i comes from ``target_rows``;
    False means it comes from ``cond_rows``. The pointer into each stream is
    incremented in order.
    """
    n_total = int(update_mask.shape[0])
    parts = []
    ci, ti = 0, 0
    # Group contiguous runs of the same mask value so we can slice+concat
    mask_bool = np.asarray(update_mask).astype(bool)
    i = 0
    while i < n_total:
        j = i + 1
        while j < n_total and mask_bool[j] == mask_bool[i]:
            j += 1
        run_len = j - i
        if mask_bool[i]:
            parts.append(target_rows[ti:ti + run_len])
            ti += run_len
        else:
            parts.append(cond_rows[ci:ci + run_len])
            ci += run_len
        i = j
    return mx.concatenate(parts, axis=0) if parts else target_rows
