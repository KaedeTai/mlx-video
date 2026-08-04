"""Phase 6 full-model load smoke test: ensure the converted 33B DiT loads
into MiniMaxH3Model without missing/extra keys."""

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_video.models.minimax_h3 import MiniMaxH3Config, MiniMaxH3Model


def test_dit_load_full():
    cfg = MiniMaxH3Config()
    m = MiniMaxH3Model(cfg)
    m.load_weights("/Users/kaede/mlx-video/mlx-models/MiniMaxH3-Ref2VA-MLX-bf16/dit/model.safetensors")
    kv = dict(tree_flatten(m.parameters()))
    # Total param count
    total = sum(v.size for v in kv.values())
    print(f"loaded {len(kv)} tensors, total params: {total:,}")
    assert total > 30_000_000_000, f"expected >30B params, got {total:,}"
    # Sanity: fp32 island preserved
    assert kv["video_patch_proj.weight"].dtype == mx.float32
    assert kv["final_layer.video_out.weight"].dtype == mx.float32
    assert kv["blocks.0.attn.qkv_proj.weight"].dtype == mx.bfloat16


if __name__ == "__main__":
    test_dit_load_full()
    print("OK")
