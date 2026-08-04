# MiniMax H3 Video VAE — Architecture Notes (Phase 2 Port)

**Reference:** `/tmp/h3_recon/ComfyUI/comfy/ldm/minimax/vae.py` (694 LOC, native
implementation) + `~/models/MiniMax-H3-raw/Ref2VA/video_vae/*.py` (12 py files
that ship with the checkpoint, more feature-complete but with training / SP /
distributed scaffolding).

**Checkpoint:** `~/models/MiniMax-H3-raw/Ref2VA/video_vae/source/model.safetensors`
- 560 keys, ~2.60 B params, stored fp32 in the source file.
- Top-level prefixes: `encoder.*` (116), `decoder.*` (440), `quant_conv.*` (2),
  `post_quant_conv.*` (2).
- Note: `AutoencoderKLLegacy` = the actual module the state-dict describes. The
  outer `MiniMaxH3VideoVAE` (in `minimax_h3_video_vae.py`) is a `.model`
  wrapper; the safetensors file stores the *inner* legacy VAE keys, unprefixed.

## Config (from `source/config.json`)

| key                  | value                       |
|----------------------|-----------------------------|
| in_channels          | 3                           |
| out_ch               | 3                           |
| ch                   | 128                         |
| embed_dim            | 24                          |
| z_channels           | 24                          |
| ch_mult              | (1, 2, 2, 4, 4, 8)          |
| num_res_blocks       | 2                           |
| space_down           | (2, 2, 2, 2, 1, 1)          |
| time_down            | (1, 2, 2, 1, 1, 1)          |
| padding_mode         | "reflect"                   |
| padding_mode_t       | None → "constant" (causal)  |
| causal_encoder       | true                        |
| causal_decoder       | false                       |
| use_t_isolated_gn    | **true**                    |
| use_vit_decoder      | true                        |
| pixel_norm_type      | "imagenet"                  |
| vae_ratio            | 16   (∏ space_down)         |
| vae_ratio_t          | 4    (∏ time_down)          |

`vit_decoder_kwargs`:

| key                | value      |
|--------------------|------------|
| dim_head           | 64         |
| heads              | 32         |
| num_layers         | 36         |
| norm_type          | "rms_norm" |
| norm_affine        | true       |
| qk_norm_type       | "rms_norm" |
| qk_norm_affine     | **false**  |
| ffn_activation_fn  | "silu"     |
| ffn_use_gated      | true       |
| rope_theta         | 100.0      |
| rope_dim_ratio     | 0.75       |
| num_register_tokens| 4 (default)|

dim = heads * dim_head = **2048**
rope apply dim = int(dim_head * rope_dim_ratio) = **48**
patchify shape (t, h, w) = (**4, 16, 16**) → 3 * 4 * 16 * 16 = 3072 per patch

## Class hierarchy

```
MiniMaxH3VideoVAE                        (top wrapper — no learned params of its own,
│                                         adds pixel normalize + latent normalize
│                                         + tiling / temporal chunking)
├── encoder: EncoderFCN3D                (3D causal CNN, 116 params)
│   ├── conv_in: CausalConv3d(3 → 128)
│   ├── down: ModuleList[6]
│   │   ├── [0] block[0..1]: ResnetBlock3D(128 → 128)
│   │   │       downsample: Downsample3D(128 → 128, ts=1, ss=2)
│   │   ├── [1] block[0..1]: ResnetBlock3D(128/256 → 256)
│   │   │       downsample: Downsample3D(256 → 256, ts=2, ss=2)
│   │   ├── [2] block[0..1]: ResnetBlock3D(256 → 256)
│   │   │       downsample: Downsample3D(256 → 256, ts=2, ss=2)
│   │   ├── [3] block[0..1]: ResnetBlock3D(256/512 → 512)
│   │   │       downsample: Downsample3D(512 → 512, ts=1, ss=2)
│   │   ├── [4] block[0..1]: ResnetBlock3D(512 → 512)
│   │   │       (no downsample: ss*ts == 1)
│   │   └── [5] block[0..1]: ResnetBlock3D(512/1024 → 1024)
│   │           block[0].nin_shortcut: CausalConv3d(512 → 1024, 1×1×1)
│   │           (no downsample)
│   ├── norm_out: GroupNorm3D(1024, 32 groups, t-isolated)
│   └── conv_out: CausalConv3d(1024 → 48)   (2 * z_channels since double_z)
│
├── quant_conv:      Conv3d(48 → 48, 1×1×1)   (2*z → 2*embed; here z=embed=24)
├── post_quant_conv: Conv3d(24 → 24, 1×1×1)
│
└── decoder: ViT3DDecoder                (36-layer ViT, 440 params)
    ├── x_embedder:      Linear(24 → 2048)
    ├── register_tokens: Parameter[1, 4, 2048]
    ├── mask_token:      Buffer[1, 1, 2048]  (unused at inference)
    ├── pos_embed:       RotaryEmbeddingND(48, base=100.0, n_dim=3, angle_scale=2π)
    ├── transformer_blocks: ModuleList[36] × TransformerBlock
    │   ├── norm1:  RMSNorm(2048)                        [weight only]
    │   ├── attn:   Attention(heads=32, dim_head=64)
    │   │   ├── norm_q: RMSNorm(64, affine=False)        [no params]
    │   │   ├── norm_k: RMSNorm(64, affine=False)        [no params]
    │   │   ├── to_qkv: Linear(2048 → 6144)              [weight + bias]
    │   │   └── to_out: Linear(2048 → 2048)              [weight + bias]
    │   ├── scale1: Parameter[2048]
    │   ├── norm2:  RMSNorm(2048)                        [weight only]
    │   ├── ff:     FeedForward (SwiGLU)
    │   │   ├── w1: Linear(2048 → 16384)                 [weight + bias]  (gate,x)
    │   │   └── w2: Linear(8192 → 2048)                  [weight + bias]
    │   └── scale2: Parameter[2048]
    ├── norm_out: LayerNorm(2048, affine=True)           [weight + bias]
    └── proj_out: Linear(2048 → 3072)                    [weight + bias]
        (out_ch=3, patch_t=4, patch=16 → 3*4*16*16=3072)
```

## Shape flow

### Encoder (single "clip" of 17 frames @ 384×384)

```
input frames    x:  (B, 3, 17, 384, 384)     pixel ∈ [-1, 1]
after pixel-normalize:                        (x+1)/2 → (val-mean)/std
conv_in         →  (B, 128, 17, 384, 384)    CausalConv3d k=3 p=1
down[0].block   →  (B, 128, 17, 384, 384)    2× ResnetBlock3D
down[0].dnsmpl  →  (B, 128, 17, 192, 192)    space 2, time 1
down[1].block   →  (B, 256, 17, 192, 192)
down[1].dnsmpl  →  (B, 256,  9,  96,  96)    space 2, time 2 (causal:  T'=ceil(17/2)=9)
down[2].block   →  (B, 256,  9,  96,  96)
down[2].dnsmpl  →  (B, 256,  5,  48,  48)    space 2, time 2
down[3].block   →  (B, 512,  5,  48,  48)
down[3].dnsmpl  →  (B, 512,  5,  24,  24)    space 2, time 1
down[4].block   →  (B, 512,  5,  24,  24)
down[5].block   →  (B,1024,  5,  24,  24)    channel change via nin_shortcut
norm_out+silu   →  (B,1024,  5,  24,  24)
conv_out        →  (B,  48,  5,  24,  24)
quant_conv      →  (B,  48,  5,  24,  24)    moments (mean + logvar); we only use mean
mean chunk      →  (B,  24,  5,  24,  24)
                                    ▲   ▲     H_lat = 384/16 = 24
                                    │   └─ vae_ratio = 16
                                    └─ T_lat = ceil(17/4) = 5
temporal chunking: token_drop=3 → keep first (5-3)=2 tokens per full-video clip
```

### Decoder (per clip: T_lat=5, H_lat=24, W_lat=24)

```
input latents  z:  (B, 24, 5, 24, 24)   after latent-denormalize (z*std + mean)
post_quant_conv →  (B, 24, 5, 24, 24)
                    permute+flatten to token seq
x_embedder      →  (B, 5*24*24=2880, 2048)   Linear(24, 2048)
concat suffix   →  (B, 2880 + 4 (reg) + 1 (cls placeholder) = 2885, 2048)
pos ids         →  create_token_ids((5,24,24), length_normalized) + zeros for suffix
rope table      →  (cos, sin), each shape (B, 2885, 1, 48)
36× TransformerBlock:
     h = h + scale1 * attn(rms_norm(h), rope)
     h = h + scale2 * ff(rms_norm(h))
norm_out        →  (B, 2885, 2048)  LayerNorm
proj_out        →  (B, 2885, 3072)  Linear
trim suffix     →  (B, 2880, 3072)
unpatchify      →  (B, 3, 5*4=20, 24*16=384, 24*16=384)
denormalize     →  (val*std + mean), clamp[0,1], to [-1,1]
```

## RMSNorm / LayerNorm parity

- PyTorch `nn.RMSNorm(dim, elementwise_affine=True)` stores only `.weight`, no bias.
- PyTorch `nn.LayerNorm(dim, elementwise_affine=True)` stores both `.weight` and `.bias`.
- Checkpoint shows `norm1.weight` and `norm2.weight` (no bias) → **RMSNorm** (per config).
- Checkpoint shows `norm_out.weight` and `norm_out.bias` → **LayerNorm** for the top norm.
- `attn.norm_q` / `attn.norm_k`: **no keys** in checkpoint (qk_norm_affine=False, so
  RMSNorm with no learnable weight — pure normalization).

## Rotary embedding details

`RotaryEmbeddingND(48, rotary_base=100.0, n_dim=3, use_angle=True)`:

- `angle_scale = 2π` (because `use_angle=True`).
- `inv_freq` shape: `1 / 100.0 ** arange(0, 1, step=2*3/48=0.125)` → shape (8,)
  (48 rot dim / 2 rotate-half / 3 axes = 8 freq pairs per axis).
- For each token `img_ids[b, n]` = 3-D coord in [-1, 1]:
    `angles[b, n, axis, freq] = 2π * coord[axis] * inv_freq[freq]`  → shape (B, N, 3, 8)
  flatten axes→freq: (B, N, 24), then `tile(2)` (repeat to 48) → (B, N, 48).
  Insert head dim: (B, N, 1, 48). Take `cos`, `sin`.
- `apply_rotary_pos_emb(t, (cos, sin))`:
    `rotate_half(t) = concat(-t[..., d/2:], t[..., :d/2])`  along last dim.
    `t_rot = t * cos + rotate_half(t) * sin` for first `rot_dim`; pass-through the rest.

The ComfyUI version stores a "table" of rotation 2×2 blocks and uses
`apply_rope_split_half`; equivalent math, different data layout. We follow the
`Ref2VA` (cos, sin) form because it matches the *stored* buffer semantics and
keeps the port simpler.

## Padding modes

- Encoder is causal (`causal=True`, `causal_encoder=True`).
- `CausalConv3d` in this VAE:
    - spatial padding: **reflect** (from config `padding_mode="reflect"`).
    - temporal padding: `padding_mode_t=None` → falls back to `"constant"` (zeros)
      because `causal=True`. **Front-only** temporal padding of size `2 * pad_t`
      when D > 1; when D == 1, the causal front-pad is all-zero so we short-circuit
      into a "causal_zero" mode that just runs a plain conv on the padded tensor.
- MLX `mx.pad` supports only `constant` and `edge`. We implement reflect padding
  ourselves via slice-and-flip along H, W (spatial) axes.

## Weight naming (1:1 with source safetensors)

We name MLX modules identically so that after transposing Conv3d weight layout,
the state dict maps element-by-element. Layout differences:

| module   | PyTorch weight shape        | MLX weight shape           | transform            |
|----------|-----------------------------|----------------------------|----------------------|
| Conv3d   | (O, I, D, H, W)             | (O, D, H, W, I)            | permute (0,2,3,4,1)  |
| Linear   | (O, I)                      | (O, I)                     | none                 |
| RMSNorm  | (D,)                        | (D,)                       | none                 |
| LayerNorm| weight (D,), bias (D,)      | weight (D,), bias (D,)     | none                 |
| Params (register_tokens, scale1, scale2, mask_token) | (1, N, D) / (D,) | same | none |

## Non-obvious quirks

1. `use_t_isolated_gn=True`: GroupNorm3D reshapes (B,C,T,H,W) → (B*T,C,1,H,W)
   so statistics are computed **per frame**, not per (T,H,W) volume.
2. `time_down = (1, 2, 2, 1, 1, 1)` means only levels 1 and 2 downsample temporally.
   Combined with the causal front-pad (size `k-1 = 2`), a 17-frame input becomes
   `ceil(17/4) = 5` latent tokens; the last `token_drop = 3` are trimmed at
   `encode_temporal`.
3. `mask_token`: stored as a *buffer* but written to safetensors anyway.
   Present in ckpt keys, but not used at inference — we allocate and load it
   for state-dict parity but never read it during forward.
4. The suffix consists of `register_tokens` (4) plus a single zero placeholder
   for what used to be a `cls_token` in the encoder — the decoder never has a
   learned cls but pads a zero anyway so its sequence length matches the
   encoder-symmetric layout.
5. `_vit_norm_input` upcasts the residual stream to fp32 before every norm
   call in the reference; MLX doesn't need this trick because our matmuls
   already accumulate in a wider type, but we still upcast for parity.

## Tiling / temporal chunking

Both `tiled_encode/decode` (spatial) and `encode_temporal/decode_temporal`
(temporal) mirror the ComfyUI reference logic 1:1. Phase 2 keeps them but
tests only the single-clip fast path (no tiling, no multi-chunk); we defer
correctness sweeps on the tiling boundaries to Phase 4 when we run
end-to-end video generation.
