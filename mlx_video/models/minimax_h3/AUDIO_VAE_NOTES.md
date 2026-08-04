# MiniMax H3 Audio VAE — Architecture Notes (Phase 3 Port)

**Reference:** `/tmp/h3_recon/ComfyUI/comfy/ldm/minimax/audio_vae.py` (443 LOC) +
`~/models/MiniMax-H3-raw/Ref2VA/audio_vae/*.py` (bundled inference module the
checkpoint's `auto_map` points at).

**Checkpoint:** `~/models/MiniMax-H3-raw/Ref2VA/audio_vae/model.safetensors`
- 1087 keys, 577 MB fp32.
- Top-level prefixes: `encoder.*`, `pre_block.*`, `mean_proj.*`, `logs_proj.*`,
  `dec_in_proj.*`, `decoder.*`.
- **172 `.weight_g`/`.weight_v` pairs** across encoder + decoder convs.
- No `latents_mean`/`latents_std` buffers in the safetensors — they live in
  `config.json` as fixed lists (`vae_latent_channels = 32` entries each).

## Config (from checkpoint's `metadata.json` + `config.json`)

| key                  | value            |
|----------------------|------------------|
| sample_rate          | 32000            |
| encoder_dim          | 64               |
| encoder_rates        | (2, 4, 4, 5, 5)  |
| latent_dim           | 2048             |
| decoder_dim          | 1024             |
| decoder_rates        | (5, 5, 2, 2, 2, 2, 2) |
| vae_latent_channels  | 32               |
| attn_proj            | true             |
| decoder_type         | "bigvgan"        |
| output_channel       | 2 (stereo)       |

**Derived constants:**
- `hop_length = ∏ encoder_rates = 2 * 4 * 4 * 5 * 5 = 800` samples/latent frame
- `latents_per_second = 32000 / 800 = 40`
- BigVGAN final channel: `1024 / 2**7 = 8`

## Class hierarchy

```
MiniMaxH3AudioVAE                             (top; register buffers latents_mean/std)
├── encoder: Encoder                          (DAC waveform encoder, 1D)
│   ├── block.0:   WNConv1d(1 → 64, k=7, p=3)                        [weight_g/v]
│   ├── block.1:   EncoderBlock(64  → 128, stride=2)
│   ├── block.2:   EncoderBlock(128 → 256, stride=4)
│   ├── block.3:   EncoderBlock(256 → 512, stride=4)
│   ├── block.4:   EncoderBlock(512 → 1024,stride=5)
│   ├── block.5:   EncoderBlock(1024→ 2048,stride=5)
│   ├── block.6:   Snake1d(2048)                                     [alpha (1,2048,1)]
│   └── block.7:   WNConv1d(2048 → 2048, k=3, p=1)                   [weight_g/v]
│       Every EncoderBlock.block = [Res1, Res3, Res9, Snake1d(dim/2), WNConv1d(dim/2 → dim, k=2s, s=s, p=ceil(s/2))]
│       Every ResidualUnit.block = [Snake1d, WNConv1d(k=7 dilated), Snake1d, WNConv1d(k=1)]
│
├── pre_block: AttnProjection(in_dim=2048, out_dim=32, num_heads=8, mlp_ratio=2)
│   ├── norm1:  LayerNorm(2048)   [w+b]
│   ├── norm3:  LayerNorm(2048)   [w+b]
│   ├── norm2:  LayerNorm(32)     [w+b]
│   ├── proj:   Linear(2048 → 32) [w+b]                       (residual projection)
│   ├── attn:   CausalAttention   (in_dim=2048, out_dim=32, heads=8)
│   │   ├── qkv:  Linear(2048 → 6144) NO bias                        [w only]
│   │   ├── q_bias, v_bias: Parameter[2048]
│   │   ├── zero_k_bias:    buffer  [2048]  (in checkpoint, forever zero)
│   │   └── proj: Linear(32 → 32) [w+b]
│   └── mlp: GeGluMlp(in=32, hidden=64)
│       ├── norm: LayerNorm(32) [w+b]
│       ├── w0:   Linear(32 → 64)  [w+b]
│       ├── w1:   Linear(32 → 64)  [w+b]
│       └── w2:   Linear(64 → 32)  [w+b]
│
├── mean_proj:   Conv1d(32 → 32, k=1)   [w+b, NO weight_norm]
├── logs_proj:   Conv1d(32 → 32, k=1)   [w+b, unused at inference]
├── dec_in_proj: Conv1d(32 → 2048, k=1) [w+b, NO weight_norm]
│
└── decoder: BigVGAN(num_mels=2048, upsample_initial_channel=1024, ...)
    ├── conv_pre:  WNConv1d(2048 → 1024, k=7, p=3)             [weight_g/v]
    ├── ups: ModuleList[7]  each = ModuleList[1] × ConvTranspose1d
    │   ├── ups.0: 1024 → 512  k=9 s=5 p=(9-5)//2 = 2         [WN — weight_g/v]
    │   ├── ups.1: 512  → 256  k=9 s=5 p=2                    [WN]
    │   ├── ups.2: 256  → 128  k=4 s=2 p=1                    [WN]
    │   ├── ups.3: 128  → 64   k=4 s=2 p=1                    [WN]
    │   ├── ups.4: 64   → 32   k=4 s=2 p=1                    [WN]
    │   ├── ups.5: 32   → 16   k=4 s=2 p=1                    [WN]
    │   └── ups.6: 16   → 8    k=4 s=2 p=1                    [WN]
    ├── resblocks: ModuleList[21]  = 7 levels × 3 kernel-sizes
    │       Each AMPBlock1(ch, k∈{3,7,11}, dil=(1,3,5)):
    │       - convs1: 3× WNConv1d(ch, ch, k, dil={1,3,5}, p=get_padding(k,d))
    │       - convs2: 3× WNConv1d(ch, ch, k, dil=1,     p=get_padding(k,1))
    │       - activations: 6× Activation1d(SnakeBeta(ch, alpha_logscale=True))
    │       Per-level ch = 512, 256, 128, 64, 32, 16, 8
    ├── activation_post: Activation1d(SnakeBeta(8, alpha_logscale=True))
    └── conv_post: WNConv1d(8 → 1, k=7, p=3, bias=False)        [weight_g/v only]
```

## Snake activations

Two flavors sit side-by-side in the checkpoint.

### Snake1d (encoder side)
- Parameter: `alpha` shape `(1, C, 1)` (per-channel, positional in NCL).
- Formula (Descript DAC): `x + (1/alpha) * sin(alpha * x)^2`.
- **NOT log-scaled** — raw value used directly.
- ComfyUI's helper `snake(x, alpha, beta)` implements `x + (1/beta) * sin(alpha*x)^2`
  and Snake1d passes the same tensor for both, i.e. `alpha == beta`. Equivalent to
  the reference DAC formulation.

### SnakeBeta (decoder side, BigVGAN)
- Parameters: `alpha` and `beta`, both shape `(C,)`.
- **Log-scaled** (`alpha_logscale=True` in the 32 kHz BigVGAN preset), so at
  forward time `alpha_eff = exp(alpha)`, `beta_eff = exp(beta)`.
- Formula: `x + (1/beta_eff) * sin(alpha_eff * x)^2`.

MLX has no built-in Snake; we implement both classes ourselves.

## Anti-aliased activation (Activation1d)

Every SnakeBeta call is wrapped in `upsample(2x) → snakebeta → downsample(2x)` with
Kaiser-windowed sinc low-pass filters (junjun3518/alias-free-torch).
- Filters live in the checkpoint as buffers of shape `(1, 1, 12)`:
  `activations.i.upsample.filter` and `activations.i.downsample.lowpass.filter`.
- `UpSample1d`: replicate-pad → grouped `conv_transpose1d(stride=2)` → crop.
- `DownSample1d`: replicate-pad → grouped `conv1d(stride=2)`.
- All three ratios are 2, kernel_size 12, `pad_left = 5`, `pad_right = 6` for LPF;
  UpSample uses `pad = 12//2 - 1 = 5`, `pad_left = 5*2 + (12-2)//2 = 15`,
  `pad_right = 5*2 + (12-2+1)//2 = 15`.

The filters are *deterministic* (fully specified by `cutoff=0.25, half_width=0.3,
kernel=12`) but the checkpoint stores them anyway; we honor stored values on load.

## Weight-norm folding

Every WNConv1d / WNConvTranspose1d stores:
- `weight_g` of shape `(O, 1, 1)`
- `weight_v` of shape `(O, I, K)`  (same as the underlying plain weight)

Fold at convert time:

```python
w = g * v / ||v||_2   # ||·|| taken over ALL dims except dim 0 (out channels)
```

That gives a plain `weight` of the same shape as `weight_v`. 172 pairs total —
same layout works for `ConvTranspose1d`.

## MLX layout translations

| Source (PyTorch)                | MLX                              | Transform          |
|---------------------------------|----------------------------------|--------------------|
| `Conv1d.weight (O, I, K)`       | `(O, K, I)`                      | permute (0, 2, 1)  |
| `ConvTranspose1d.weight (I, O, K)` | `(O, K, I)`  (flip in/out)    | permute (1, 2, 0)  |
| `Linear.weight (O, I)`          | `(O, I)`                         | none               |
| `LayerNorm.weight/bias (D,)`    | `(D,)`                           | none               |
| filter buffers `(1, 1, K)`      | expanded per group at conv time  | keep as (1, 1, K); expand+permute at forward |
| `alpha (1, C, 1)` (Snake1d)     | keep as `(1, C, 1)`              | none               |
| `alpha/beta (C,)`  (SnakeBeta)  | keep as `(C,)`                   | none               |

**Note on grouped conv semantics.** For the anti-alias filters we need
`groups = C` in a 1D conv. MLX's `mx.conv_general` supports `groups`; the
per-group weight shape is `(O=C, K, I/groups=1)`. We compute the broadcast
`filter.expand(C, -1, -1) → (C, 1, K)` and permute to `(C, K, 1)` at call time.

## Data layout convention

MLX's Conv1d expects **NLC** (batch, length, channels). The reference uses
**NCL**. We do a `swapaxes(1, 2)` on the audio boundary of every conv-based
class. `encode` / `decode` take/return tensors in the reference's NCL layout to
keep the module signature source-compatible.

## Shape flow (encoder, 5s stereo @ 32 kHz)

```
input   waveform:  (1, 2, 160000)                    stereo NCL
reshape → mono:    (2, 1, 160000)                    b*s treated as batch
                   (padded up to multiple of 800 samples)
encoder.block.0:   (2, 64,   160000)   WNConv1d k=7 p=3, stride=1
encoder.block.1:   (2, 128,   80000)   EncoderBlock stride 2
encoder.block.2:   (2, 256,   20000)   stride 4
encoder.block.3:   (2, 512,    5000)   stride 4
encoder.block.4:   (2, 1024,   1000)   stride 5
encoder.block.5:   (2, 2048,    200)   stride 5
encoder.block.6-7: (2, 2048,    200)   final Snake1d + WNConv1d k=3
pre_block(swap):   (2, 200, 32)        Attn projection to latent width
mean_proj:         (2, 32, 200)        Conv1d 1x1
normalize by latents_mean/std:         (2, 32, 200)
reshape → stereo:  (1, 32, 2, 200)     [B, C_lat, S=2, T=200] = 40 fps × 5s
```

## Notable inference quirks

1. **encode returns the posterior mean** — no sampling, no `logs_proj` call.
2. **decode clamps to [-1, 1]** — `use_tanh_at_final = False`, so a plain
   `mx.clip(x, -1, 1)` after `conv_post`.
3. **Latents_mean / latents_std** come from `config.json` (32 fp64 numbers each),
   not from the safetensors. We hard-code them into the MLX module and register
   them as float arrays.
4. **Encoder crops residuals in ResidualUnit** — when the k=7 dilated conv shrinks
   the length, the skip connection is center-cropped to match (`x[..., pad:-pad]`).
5. **AttnProjection is asymmetric** — 2048 → 32. The attn branch mean-pools over
   heads then adaptive_avg_pool1d down to 32. The parallel residual is a straight
   Linear(2048 → 32) applied to a separately-normed copy of the input.
6. **BigVGAN sums k-parallel AMPBlocks** then divides by `num_kernels = 3`.
7. **conv_post has no bias.**
