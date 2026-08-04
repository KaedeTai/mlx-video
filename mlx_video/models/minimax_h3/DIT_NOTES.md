# H3 DiT — MLX port notes (Phase 4)

## Files

| File | LOC | Reference | Notes |
|---|---|---|---|
| `rope.py` | 100 | `model.py:130-140,508-518` | 3-axis split-half rotary |
| `attention.py` | 118 | `model.py:141-174` | RMSNorm+rope+MHA, plain-MLX (no fused kernel) |
| `packed_layout.py` | 208 | `model.py:281-378` | numpy-only PackedLayout |
| `blocks.py` | 231 | `model.py:114-279` | TimeEmbedder / AdalnProj / MLP / RefinerBlock / DiTBlock / FinalLayer |
| `model.py` | 320 | `model.py:380-646` | MiniMaxH3Model + patchify helpers |

Total: ~977 LOC of MLX (vs 646 LOC of PyTorch reference — some overhead from
functional slice/concat instead of in-place ops).

## Key adaptations vs the reference

### 1. In-place → functional
PyTorch uses `.add_`, `.mul_`, `.addcmul_` on the residual stream. MLX arrays
are immutable, so `_mod_scale_shift` / `_mod_gate` walk the (contiguous)
segment list, slice the stream, modulate each piece, and concatenate. This is
correct because segments are guaranteed to cover `[0, seq_len)` without gaps.

### 2. Segment-indexed adaLN → slice+concat
Reference:
```python
for a, b, row in segments:
    h[a:b].mul_(1.0 + scale[row]).add_(shift[row])
```
MLX:
```python
parts = [h[a:b] * (1.0 + scale[row:row+1]) + shift[row:row+1] for a, b, row in segments]
return mx.concatenate(parts, axis=0)
```

### 3. Split-half rope on plain MLX
The reference uses `comfy.quant_ops.ck.rms_rope_split_half` (fused CUDA/MPS
kernel). We compute the rotation table once per forward (`build_rope_table`),
then apply the 2×2 rotation matrix over `(x_a, x_b)` pairs where
`x_a = x[..., :half]`, `x_b = x[..., half:rot_dim]`. The tail `x[..., rot_dim:]`
passes through un-rotated. This is a ~1.5-2× slowdown vs the fused kernel; a
Metal shim is a Phase 8 optimization.

### 4. Row weaving via mask
For ref2va, conditioning frames and target frames are interleaved along the
image row axis according to `img_update`. `_weave_by_mask` groups contiguous
mask runs and slices/concats each stream — avoids item assignment.

### 5. fp32 output heads
`FinalLayer.{video,audio}_out` are stored fp32 in the checkpoint. We cast the
modulated activation to fp32 before applying them (matches reference).

### 6. Row-major stream assembly
Reference builds `h` via item assignment: `h[a:b] = text_states` etc.
MLX has no item assignment. We instead append slices to a list in segment
order and `mx.concatenate(parts, axis=0)` at the end.

## Weight naming (1:1 with Ref2VA)

Verified via `mlx.utils.tree_flatten(model.parameters())`:
- `video_patch_proj.{weight,bias}` — fp32 patch projector
- `audio_patch_proj.{weight,bias}` — fp32 patch projector
- `condition_proj.{weight,bias}` — text→hidden refiner
- `time_embedder.proj_{in,out}.{weight,bias}` — sinusoidal + MLP
- `rope.inv_freq` — `[16]` (fp32)
- `token_refiner.blocks.N.*` + `token_refiner.final_norm.weight` — 2-block text refiner
- `blocks.N.norm{1,2}.weight` — RMSNorm per-block
- `blocks.N.attn.{qkv_proj,q_norm,k_norm,out_proj}.weight`
- `blocks.N.mlp.{fc1,fc2}.weight` — SwiGLU
- `blocks.N.adaln_proj.linear.{weight,bias}` — `[6*hidden*3, t_dim]` per block
- `final_layer.norm.weight`
- `final_layer.adaln_proj.linear.{weight,bias}` — `[2*hidden*1, t_dim]`
- `final_layer.{video_out,audio_out}.{weight,bias}` — fp32 heads

## Smoke test (`tests/test_h3_dit_smoke.py`)

Three tests, all passing:
1. `test_rope_math` — build a rotation table for 10 positions, verify shape
   `[1, 10, 1, 48, 2, 2]`, apply to `[10, 4, 128]` q tensor, verify tail
   `x[..., 96:]` passes through unchanged.
2. `test_packed_layout_t2va` — build a `PackedLayout(text=5, latent_t=2, 8×8,
   audio_t=3)`, verify segment kinds and `seq_len == 5 + 6 + 32`.
3. `test_dit_forward_tiny` — build a tiny DiT (hidden=192, 2 blocks, heads=3,
   head_dim=64), run one forward, verify output shapes match inputs and are
   finite.

Full-config numerical parity vs ComfyUI reference is a Phase 6 test (needs
loaded weights).

## Known limitations (defer to Phase 8)

- No fused RMSNorm+rope kernel — ~1.5-2× slower than reference.
- No `model_prefetch` streaming — the full 29 GB bf16 DiT is held resident.
- No `optimized_attention` varlen packed path — we use `mx.fast.sdpa` on the
  full packed sequence (fine for a single batch).
- Curve-basis adaLN branch (`use_adaln_curves`) is implemented but untested;
  Ref2VA doesn't use it.
