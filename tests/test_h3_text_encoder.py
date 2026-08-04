"""Phase 8-1: real Qwen3-VL-32B-4bit truncated-at-50 text encoder wiring."""
import os
import numpy as np
import pytest
import mlx.core as mx

QWEN_PATH = os.path.expanduser("~/models/Qwen3-VL-32B-Instruct-4bit")
pytestmark = pytest.mark.skipif(not os.path.isdir(QWEN_PATH),
                                reason="Q4 Qwen3-VL not local")


def test_shape_and_determinism():
    from mlx_video.models.minimax_h3.text_encoder_bridge import TextEncoderBridge
    enc = TextEncoderBridge(QWEN_PATH, truncate_layer=50)
    a = np.asarray(enc.encode("hello world"))
    b = np.asarray(enc.encode("hello world"))
    assert a.shape[0] == 1 and a.shape[2] == 5120
    assert np.allclose(a, b)


def test_prompt_conditional():
    from mlx_video.models.minimax_h3.text_encoder_bridge import TextEncoderBridge
    enc = TextEncoderBridge(QWEN_PATH, truncate_layer=50)
    a = np.asarray(enc.encode("a happy dog"))
    b = np.asarray(enc.encode("a stormy sky"))
    # Different prompts must differ in raw hidden state
    # (mean-pooled cosine may be high; we compare raw L2 per-position after align)
    n = min(a.shape[1], b.shape[1])
    diff = np.abs(a[:, :n] - b[:, :n]).mean()
    assert diff > 0.5, f"encoder output too similar across prompts (diff={diff})"


def test_truncation():
    from mlx_video.models.minimax_h3.text_encoder_bridge import TextEncoderBridge
    enc = TextEncoderBridge(QWEN_PATH, truncate_layer=50)
    assert len(enc._lang.model.layers) == 50
    assert enc.text_dim == 5120
