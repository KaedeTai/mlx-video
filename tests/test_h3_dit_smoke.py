"""Phase 4 smoke test: build the DiT with random weights on a tiny config and
verify one forward pass produces sane shapes/finite values.

We use a stripped-down config (smaller hidden, few layers) so this runs in
seconds. Full-weight numerical parity vs ComfyUI is a separate test.
"""

import numpy as np
import mlx.core as mx

from mlx_video.models.minimax_h3.config import MiniMaxH3Config
from mlx_video.models.minimax_h3.model import MiniMaxH3Model
from mlx_video.models.minimax_h3.packed_layout import PackedLayout
from mlx_video.models.minimax_h3.rope import build_rope_table, apply_split_half_rope


def test_rope_math():
    inv_freq = np.arange(1, 17, dtype=np.float32) / 16.0
    pos = np.zeros((10, 3), dtype=np.float64)
    pos[:, 0] = np.arange(10)
    tbl = build_rope_table(pos, inv_freq, dtype=mx.float32)
    assert tbl.shape == (1, 10, 1, 48, 2, 2), tbl.shape
    x = mx.random.normal((10, 4, 128))
    y = apply_split_half_rope(x, tbl, rot_dim=96)
    assert y.shape == (10, 4, 128), y.shape
    # Un-rotated tail passes through unchanged
    assert mx.allclose(y[..., 96:], x[..., 96:], atol=1e-5)


def test_packed_layout_t2va():
    layout = PackedLayout(text_len=5, latent_t=2, latent_h=8, latent_w=8, audio_t=3)
    # Segments: text + audio + video
    kinds = [s[2] for s in layout.segments]
    assert kinds == ["text", "audio", "video"]
    assert layout.seq_len == 5 + 3 * 2 + 2 * (8 // 2) * (8 // 2)
    assert layout.position_ids.shape == (layout.seq_len, 3)


def test_dit_forward_tiny():
    # Small config to keep smoke fast (~2 seconds on CPU/MLX)
    cfg = MiniMaxH3Config(
        hidden_size=192,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=3,
        attention_head_dim=64,
        ffn_hidden_size=256,
        latents_dim=24,
        audio_latents_dim=32,
        patch_size=(1, 2, 2),
        text_dim=192,
        timestep_input_dim=64,
        time_embed_hidden_size=192,
        time_embed_dim=96,
        rope_inv_freq_len=8,
    )
    model = MiniMaxH3Model(cfg)
    # Populate rope.inv_freq with something non-zero (matches diffusers default)
    freq = np.arange(1, cfg.rope_inv_freq_len + 1, dtype=np.float32) / cfg.rope_inv_freq_len
    model.rope.inv_freq = mx.array(freq)
    mx.eval(model.parameters())

    B, C_v, T, H, W = 1, cfg.latents_dim, 2, 8, 8
    C_a, ch, aT = cfg.audio_latents_dim, 2, 4
    video = mx.random.normal((B, C_v, T, H, W)) * 0.1
    audio = mx.random.normal((B, C_a, ch, aT)) * 0.1
    context = mx.random.normal((1, 5, cfg.text_dim))
    ts = mx.array([500.0])  # sigma * 1000

    out_v, out_a = model((video, audio), ts, context, payload={})
    assert out_v.shape == video.shape, (out_v.shape, video.shape)
    assert out_a.shape == audio.shape, (out_a.shape, audio.shape)

    # Finite
    ov_np = np.asarray(out_v).astype(np.float32)
    oa_np = np.asarray(out_a).astype(np.float32)
    assert np.isfinite(ov_np).all(), "video out has NaN/inf"
    assert np.isfinite(oa_np).all(), "audio out has NaN/inf"


if __name__ == "__main__":
    test_rope_math()
    print("rope math OK")
    test_packed_layout_t2va()
    print("packed layout OK")
    test_dit_forward_tiny()
    print("dit forward tiny OK")
