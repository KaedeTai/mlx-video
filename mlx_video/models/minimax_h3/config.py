"""MiniMax H3 config.

Values default to the Ref2VA (`~/models/MiniMax-H3-raw/Ref2VA/transformer/config.json`)
5376-dim / 50-layer / 56-head DiT with (1, 2, 2) video patching and 24/32
video/audio latent channels. Curve-basis adaLN (`adaln_curve_grid`) is None for
this checkpoint; leave it unset unless a future H3 release ships an
`adaln_t_table` buffer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class MiniMaxH3Config:
    # DiT
    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336

    # Streams
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: Tuple[int, int, int] = (1, 2, 2)

    # Conditioning (Qwen3-VL-32B layer-50 hidden state)
    text_dim: int = 5120

    # Time embedding
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    # If set, replaces the time embedder with a shared curve basis of shape
    # (adaln_curve_grid, time_embed_dim). Ref2VA does NOT use this path.
    adaln_curve_grid: Optional[int] = None

    # RoPE
    rope_inv_freq_len: int = 16  # per-axis; total rot dim = 3 * 2 * 16 = 96

    # Norms
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    # Flow-matching dual sigma shift (video drives the sampler, audio is derived)
    sigma_shift_video: float = 12.0
    sigma_shift_audio: float = 3.0

    # Sampling / preset defaults (mirrors comfy_extras/nodes_minimax_h3.py)
    canvas_multiple: int = 32
    base_short_edge: int = 768
    max_pixels: int = 768 * 1344
    ref_image_short_edge: int = 2048
    fps: int = 24
    audio_latent_fps: int = 40

    @property
    def video_patch_dim(self) -> int:
        p_t, p_h, p_w = self.patch_size
        return self.latents_dim * p_t * p_h * p_w

    @property
    def rope_total_rot_dim(self) -> int:
        # 3 axes × 2 halves × inv_freq_len == rotation-table width
        return 3 * 2 * self.rope_inv_freq_len

    @classmethod
    def ref2va(cls) -> "MiniMaxH3Config":
        """Reference-to-Video+Audio checkpoint (the one we have locally)."""
        return cls()  # defaults are Ref2VA

    @classmethod
    def from_hf_config(cls, path: str | Path) -> "MiniMaxH3Config":
        """Load Ref2VA/transformer/config.json (diffusers format) and translate.

        Diffusers ships extra derived fields (`adaln_out_features`,
        `final_adaln_out_features`) that we ignore — they are computed from
        `time_embed_dim`, `hidden_size`, and `latents_dim` at build time.
        """
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for k, v in raw.items():
            if k in known:
                # patch_size arrives as a list in JSON; make it a tuple
                if k == "patch_size":
                    v = tuple(v)
                kwargs[k] = v
        return cls(**kwargs)


# Convenience for the pipeline
def load_ref2va_config(
    ref2va_root: str | Path = "~/models/MiniMax-H3-raw/Ref2VA",
) -> MiniMaxH3Config:
    root = Path(ref2va_root).expanduser()
    return MiniMaxH3Config.from_hf_config(root / "transformer" / "config.json")
