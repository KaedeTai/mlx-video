# MiniMax H3 → MLX Port Plan

**Status:** Phase 2 complete (Video VAE numerically parity-checked). Phases 3–8 not started.
**Estimated total time:** 4–6 weeks of focused work (Phase 2 done in <1 day vs 1-week estimate).
**Target device:** Apple Silicon (M-series) via MLX.

### Phase 2 completion snapshot (2026-08-04)
- Encoder parity vs PyTorch reference: **139.25 dB** on 64×64×1-frame input
- Decoder parity vs PyTorch reference: **62.99 dB** on 4×4×1-token latent
- Image round-trip vs PyTorch reference: **63.05 dB** (MLX-vs-PT agreement)
- 5f × 384×384 encode + decode: **1.88 s + 0.25 s** (bf16, cold)
- 17f × 384×384 encode + decode: **1.77 s + 0.23 s**
- Peak RSS during forward: **5.32 GB** (weights 5.0 GB + activations ~0.3 GB)
- Deliverables: `video_vae.py` (609 LOC), `convert.py` VAE section (154 LOC),
  `VIDEO_VAE_NOTES.md`, `tests/test_h3_video_vae.py` (194 LOC, 4/4 passing)

---

## 1. Overview

MiniMax H3 is a joint audio+video DiT that denoises a *single packed token stream* containing text, optional reference blocks, audio, and video, then decodes each stream through its own VAE.

- **Video stream:** 24-channel 3D latent, patch (1, 2, 2). Downscale ratio: 16 spatial, 4 temporal. VAE = 3D causal CNN encoder + ViT3D decoder (36 layers, `heads=32`, `dim_head=64`).
- **Audio stream:** 32-channel stereo (channels-major) at 40 latent frames per second. VAE = DAC-lineage encoder + BigVGAN decoder (32 kHz stereo).
- **Conditioning:** Qwen3-VL-32B **truncated to 50 layers** (its unnormalized layer-50 hidden state). NOT chat-templated: raw text with `<Picture N>`, `<Video N>`, `<Audio N>` labels and vision blocks spliced in.
- **DiT:** 50 layers × hidden 5376 × 56 heads × head_dim 128 × ffn 14336. Token refiner: 2 blocks. adaLN modulates 3 modality tags × N unique timesteps.
- **Scheduler:** Flow matching, two independent sigma shifts (video 12.0, audio 3.0). The sampler drives one flat ODE on the video schedule; the model returns the audio velocity multiplied by `d(sigma_audio)/d(sigma_video)` so the same integrator produces both streams' true ODEs.
- **RoPE:** 3D packed (t, h, w). `inv_freq_len=16` per axis → 48 pair angles → 96-wide rotation matrices, split-half.

## 2. Reference implementations we lean on

| Purpose | Path | LOC |
|---|---|---|
| DiT modeling (primary) | `/tmp/h3_recon/ComfyUI/comfy/ldm/minimax/model.py` | 646 |
| Video VAE | `/tmp/h3_recon/ComfyUI/comfy/ldm/minimax/vae.py` | 694 |
| Audio VAE | `/tmp/h3_recon/ComfyUI/comfy/ldm/minimax/audio_vae.py` | 443 |
| Text encoder wiring | `/tmp/h3_recon/ComfyUI/comfy/text_encoders/minimax.py` | 201 |
| Node / pipeline glue | `/tmp/h3_recon/ComfyUI/comfy_extras/nodes_minimax_h3.py` | 337 |
| Model detection / factory | `/tmp/h3_recon/ComfyUI/comfy/{model_base,supported_models,model_detection,latent_formats}.py` | – |
| Bundled diffusers-format modeling (audio/video VAE only) | `~/models/MiniMax-H3-raw/Ref2VA/{video_vae,audio_vae}/*.py` | – |
| Our existing MLX port to mirror | `~/mlx-video/mlx_video/models/wan_2/` | 8025 |
| Runbook / architecture notes | `~/models/MiniMax-H3-raw/RUN_REF2VA.md` | 268 |

The ComfyUI code is the single source of truth: it is (a) actively maintained, (b) already validated end-to-end against the Ref2VA safetensors, and (c) simpler than the diffusers PR draft (which is 9354 additions across 35 files).

## 3. Weight source decision

**Chosen source: original Ref2VA bf16 diffusers shards (`~/models/MiniMax-H3-raw/Ref2VA/`).**

Rationale:
- **Transformer keys map 1:1 to ComfyUI's `MiniMaxH3Model`.** All 535 keys under `blocks.N.{attn.qkv_proj,attn.{q,k}_norm,attn.out_proj,mlp.{fc1,fc2},norm1,norm2,adaln_proj.linear}`, plus `video_patch_proj`, `audio_patch_proj`, `condition_proj`, `time_embedder.{proj_in,proj_out}`, `rope.inv_freq`, `token_refiner.blocks.N.*`, `final_layer.{norm,adaln_proj.linear,video_out,audio_out}` are present in both. **No rename table needed.**
- **Video VAE weights:** Ref2VA ships two copies. The `video_vae/source/model.safetensors` is the pre-conversion original (`AutoencoderKLLegacy`). The top-level `video_vae/model.safetensors` is the "converted" form ComfyUI loads. Detection key `decoder.transformer_blocks.0.scale1` + `encoder.down.5.block.0.conv1.weight`.
- **Audio VAE weights:** Ref2VA `audio_vae/model.safetensors` still has `weight_g`/`weight_v` weight-norm parametrizations. ComfyUI folds these into plain `weight` at load time (see `audio_vae.py` header comment: "converted checkpoint (plain '*.weight' tensors) with strict=True"). Our `convert.py` will fold `weight = weight_g * (weight_v / ||weight_v||)` and mirror the audio VAE's plain-`weight` naming.
- **Quantized `MiniMax-H3-Ref2VA-Q4/` (43 GB):** quanto qint4 groupsize-128 per-row-linear format (`.weight._data`, `.weight._scale`, `.weight._shift`). MLX has no direct quanto reader; we would need to dequantize on load, losing the size win. **Skip this source for the port; keep it only as a scratch benchmark comparison for our own MLX 4-bit later.**
- **Comfy repackaged single-file `int8_convrot`:** not present locally, and MLX has no native convrot support — dequant would need to happen on our side.

**Final loading path:** `convert.py` streams the 13 Ref2VA transformer shards + 1 video VAE `.safetensors` + 1 audio VAE `.safetensors` + 27 text encoder shards, converts each tensor bf16→float16 (Metal-native), applies audio-VAE weight-norm folding + Qwen key rewrites (`model.language_model.` → `text_model.`, and truncate to layer 50), and writes MLX-format safetensors under `~/models/MiniMax-H3-Ref2VA-mlx/`.

## 4. Weight mapping table (partial, DiT only — full mirror; audio VAE needs fold)

| ComfyUI key template | Ref2VA (diffusers) key template | Notes |
|---|---|---|
| `video_patch_proj.{weight,bias}` | `video_patch_proj.{weight,bias}` | fp32 island |
| `audio_patch_proj.{weight,bias}` | `audio_patch_proj.{weight,bias}` | fp32 island |
| `condition_proj.{weight,bias}` | `condition_proj.{weight,bias}` | bf16 |
| `time_embedder.proj_{in,out}.{weight,bias}` | same | fp32 |
| `rope.inv_freq` | same | fp32 |
| `token_refiner.blocks.N.{attn.{qkv_proj,out_proj},attn.{q,k}_norm.weight,norm1.weight,norm2.weight,mlp.{fc1,fc2}.weight}` | same | bf16 |
| `token_refiner.final_norm.weight` | same | |
| `blocks.N.attn.qkv_proj.weight` | same | bf16, no bias |
| `blocks.N.attn.out_proj.weight` | same | |
| `blocks.N.attn.{q,k}_norm.weight` | same | |
| `blocks.N.norm{1,2}.weight` | same | RMSNorm |
| `blocks.N.mlp.fc1.weight` | same | SwiGLU (fc1 out = 2*ffn) |
| `blocks.N.mlp.fc2.weight` | same | |
| `blocks.N.adaln_proj.linear.{weight,bias}` | same | (linear out = 6 × hidden × 3 modalities) |
| `final_layer.norm.weight` | same | |
| `final_layer.adaln_proj.linear.{weight,bias}` | same | (linear out = 2 × hidden × 1) |
| `final_layer.{video,audio}_out.{weight,bias}` | same | fp32 island |

**Total DiT parameters:** 535 tensors, 37 unique templates, ~29 GB in bf16.

## 5. Config (already implemented in `config.py`)

```python
MiniMaxH3Config(
    hidden_size=5376, num_layers=50, token_refiner_num_layers=2,
    num_attention_heads=56, attention_head_dim=128, ffn_hidden_size=14336,
    latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2), text_dim=5120,
    timestep_input_dim=256, time_embed_hidden_size=5376, time_embed_dim=2688,
    rope_inv_freq_len=16, norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5,
    sigma_shift_video=12.0, sigma_shift_audio=3.0,
)
```

The config also carries the `adaln_curve_grid` field (default None). If a future H3 checkpoint ships an `adaln_t_table` buffer (curve-basis adaLN), setting `adaln_curve_grid=<grid>` swaps the time embedder for interpolation over the shared basis (see `model.py` in ComfyUI, `use_adaln_curves` branch). The Ref2VA checkpoint we have does NOT use curves.

## 6. Phased plan

Each phase lists **goal, source refs, deliverables, effort, dependencies, risks.**

### Phase 2 — Video VAE port (1 week)

- **Goal:** MLX `MiniMaxH3VideoVAE.decode(z)` and `.encode(x)` that byte-match ComfyUI's on a small clip (~5-frame RGB tensor).
- **Sources:** `comfy/ldm/minimax/vae.py` (primary); `Ref2VA/video_vae/source/vae_cnn.py + vae_vit.py + minimax_h3_video_vae.py` (secondary — the pre-conversion original, useful for the ViT3D decoder's grid/patch math).
- **Deliverables:** `video_vae.py` implementing `CausalConv3d`, `TemporalIsolatedGroupNorm`, `Downsample3D`, `ResnetBlock3D`, `EncoderFCN3D` (6-level, `ch_mult=(1,2,2,4,4,8)`, `space_down=(2,2,2,2,1,1)`, `time_down=(1,2,2,1,1,1)`), `RotaryEmbeddingND` (3D), `Attention` w/ split-half rope, `TransformerBlock` w/ 2 learned per-dim scales, `ViT3DDecoder` (36 blocks × heads=32 × dim_head=64, `patch_size=16`, `patch_size_t=4`, `num_register_tokens=4`), full `MiniMaxH3VideoVAE` w/ spatial tiling (`tile_size=256`, `overlap_min=64`) + temporal chunking (`clip_length=17`, `token_drop=3`).
- **Effort:** ~5 days. VAE is heavy (694 LOC) but self-contained and testable in isolation.
- **Depends on:** nothing new (uses existing `mlx.core.conv3d`, `mx.fast.rms_norm`, `mx.fast.layer_norm`).
- **Risks:** ⚠ MLX `Conv3d` reflect-padding support (must verify — Wan2 uses causal replicate). ⚠ ViT3D's `create_token_ids` returns [-1, 1] normalized 3D coords; `RotaryEmbeddingND` uses `rotary_base=100.0` (not 10000!) and `angle_scale = 2π`. Numerically sensitive.

### Phase 3 — Audio VAE port (3–5 days)

- **Goal:** MLX `MiniMaxH3AudioVAE.decode(z)` byte-match on a synthetic 1 s stereo signal.
- **Sources:** `comfy/ldm/minimax/audio_vae.py` (primary); `Ref2VA/audio_vae/dac_*.py` (secondary — pre-conversion DAC modules).
- **Deliverables:** `audio_vae.py` implementing Snake / SnakeBeta activations, alias-free `UpSample1d`/`DownSample1d`/`LowPassFilter1d` (Kaiser-windowed sinc, `kernel_size=12`), DAC `Encoder` (`strides=(2,4,4,5,5)`, `d_latent=2048`), `AttnProjection` (`CausalAttention` w/ separate q/v biases + `zero_k_bias` buffer, GEGLU MLP), full `BigVGAN` decoder (`upsample_rates=(5,5,2,2,2,2,2)`, `resblock_kernel_sizes=(3,7,11)`), top-level `MiniMaxH3AudioVAE` (32 kHz, 40 latent fps, `samples_per_latent=800`).
- **Effort:** ~4 days.
- **Depends on:** Phase 2 (to reuse the tiling / chunking helper patterns).
- **Risks:** ⚠ MLX `ConvTranspose1d` correctness at the BigVGAN upsample edges. ⚠ `torch.kaiser_window` has no direct MLX equivalent — we precompute the filter on the numpy/CPU side in `convert.py` and register it as a buffer. ⚠ **Convert.py must fold weight-norm** (`weight = weight_g * weight_v / ||weight_v||` along the conv output-channel axis) for every `decoder.{conv_pre,conv_post,ups.*,resblocks.*.convs*}` tensor.

### Phase 4 — DiT transformer port (2–3 weeks, THE HEAVY ONE)

- **Goal:** MLX `MiniMaxH3Model.__call__(video_x, audio_x, timestep, context, payload)` numerically matching ComfyUI's `_forward` on a fixed random seed / fixed inputs.
- **Sources:** `comfy/ldm/minimax/model.py` (single file, 646 LOC — the entire DiT).
- **Modules to build:**
  - `attention.py`: fused RMSNorm+split-half-rope QKV path (ComfyUI uses `comfy.quant_ops.ck.rms_rope_split_half` — we implement a plain-MLX equivalent: apply `mx.fast.rms_norm` to q/k head-dim, then apply the rotation table on the first `rot_dim` chunk of each). Verify against ComfyUI kernel with fp32 tolerance ~1e-4.
  - `rope.py`: `rope_rotation_table(angles, dtype)` returning `[1, S, 1, half, 2, 2]` matrices (split-half convention: `angles[:, :half] == angles[:, half:]`); `PackedLayout.rope_freqs()` → the 3-axis packed position → 96-wide rope (`inv_freq_len=16` per axis × 3 axes × 2 halves).
  - `blocks.py`: `TimeEmbedder` (sin/cos, cos-before-sin), `AdalnProj` (silu → linear → chunk into `expand` per-modality tensors), `_mod_scale_shift` (in-place scale/shift over segment list), `_mod_gate` (in-place gated residual), `RefinerBlock`, `TokenRefiner`, `DiTBlock`, `FinalLayer` (fp32 output heads).
  - `packed_layout.py`: `PackedLayout` builder for t2va/fl2va (keyframe first/last) and ref2va (per-block frame grids, stereo channel-major audio grid, area-normalized (h, w) axes via `_axis_from_sqrt_area`, per-frame `FRAME_RESCALE=5/3` temporal cursor via `_video_t_spans` on `FRAME_PER_TOKEN=(1,4,4,4,4)`); yields `.position_ids [S,3] fp64`, `.segments [(a,b,kind)]`, `.img_pos` / `.img_update` masks for splicing cond vs. denoised rows.
  - `model.py`: `MiniMaxH3Model.__call__` — build `t_a` from `t_v` via `time_shift_sigma`; enumerate unique per-segment timesteps; build `mod_segments` (respecting `text_token_tags` for tag runs in the text span); patchify video with `(1,2,2)` (einsum `nctrhpwq→nthwcrpq`); pack audio channels-major; embed cond rows via `_cond_video_rows` / `_cond_audio_rows` (noise-aug seed convention: `seed` for video, `seed+1` for audio); route through 50 blocks; final layer split; unpatchify + `unpack_audio`; return `[-video_out, -slope_a * audio_out]` where `slope_a = time_shift_slope(sigma_v, shift_v, shift_a)`.
- **Deliverables:** all files above wired through `__init__.py`, exports `MiniMaxH3Model`.
- **Effort:** ~15 days.
- **Depends on:** Phases 2–3 for tiling patterns, `mlx-community` Qwen3-VL for text embedder (or Phase 6 for a stripped in-repo copy).
- **Risks:** ⚠ **In-place `.add_` / `.mul_` / `.addcmul_` semantics** — MLX arrays are immutable; every in-place op must be rewritten as functional. This is 20+ call sites. ⚠ Split-half rope + fused RMSNorm kernel — we replicate ComfyUI's op ordering (RMSNorm on q/k head-dim, then rope on `rot_dim` slice). ⚠ Segment-indexed adaLN (`mod_segments = [(a,b,row)]`) needs to translate to gather/scatter on MLX (which lacks item assignment) — consider using `mx.take` + masked stitch. ⚠ **`optimized_attention` fallback:** ComfyUI has fused kernels; we use `mx.fast.scaled_dot_product_attention` (or the manual softmax path for causal / audio-VAE cases). ⚠ **PackedLayout.signature caching** — precompute per prompt in `generate.py`, not per step.

### Phase 5 — Scheduler port (2–3 days)

- **Goal:** MLX flow-matching sampler that respects the H3 dual-shift setup.
- **Sources:** `comfy/model_sampling.ModelSamplingDiscreteFlow` + `comfy_extras/nodes_minimax_h3.py::MiniMaxH3SigmaShift` + `comfy/ldm/minimax/model.py::time_shift_sigma` / `time_shift_slope`; our own `wan_2/scheduler.py` for the flow-matching skeleton.
- **Deliverables:** `scheduler.py` with `FlowMatchScheduler(shift_video=12.0, shift_audio=3.0)`. Model returns velocity already scaled by `slope_a` for the audio stream, so the scheduler only needs the *video* schedule; it steps the packed `NestedTensor` (video, audio) simultaneously with the same sigma step. Support Euler and DPM++2M (mirror `wan_2/scheduler.py`).
- **Effort:** ~2 days.
- **Risks:** low — algebra is closed-form.

### Phase 6 — Weight convert script (3–5 days)

- **Goal:** `python -m mlx_video.models.minimax_h3.convert --src ~/models/MiniMax-H3-raw/Ref2VA --dst ~/models/MiniMax-H3-Ref2VA-mlx` produces a fully-loadable MLX bundle in <15 min on M-series.
- **Sources:** `wan_2/convert.py` (skeleton — `sanitize_wan_transformer_weights`, streaming safetensors load); ComfyUI `sd.py` VAE detection logic for VAE-specific rewrites.
- **Deliverables:** `convert.py` with:
  - `convert_transformer(src, dst)`: 13 shards → 1 file, bf16→fp16.
  - `convert_video_vae(src, dst)`: single file, no rewrite.
  - `convert_audio_vae(src, dst)`: **fold weight-norm** on every conv, drop `logs_proj` (unused at inference), drop `mask_token` (buffer, unused).
  - `convert_text_encoder(src, dst)`: import Qwen3-VL from `mlx-community/Qwen3-VL-32B-Instruct-4bit` if available; otherwise rewrite the 27 Ref2VA text_encoder shards (`model.language_model.` → `text_model.`), TRUNCATE to layers 0–49, drop `lm_head.weight` and `model.language_model.norm.weight`.
- **Effort:** ~4 days.
- **Risks:** ⚠ Peak RAM during load — stream one shard at a time. ⚠ Text encoder key rewrite for the Qwen3-VL layer-50 truncation is nontrivial.

### Phase 7 — Pipeline glue + smoke test (3–5 days)

- **Goal:** `python -m mlx_video.models.minimax_h3.generate --task ref2va --prompt ... --ref_image ... --ref_audio ... --output out.mp4` runs end-to-end and produces a video file (quality not yet validated — just: no NaN, correct shapes, output plays).
- **Sources:** `comfy_extras/nodes_minimax_h3.py` (the whole file is the pipeline recipe: canvas sizing at 768 short edge / 768×1344 area cap, `align_frame_count(n)` snap to `17k+5`, `video_latent_t`, `temporal_shape`, `_encode_ref_audio`, `_resize`, keyframe / reference block assembly); our `wan_2/generate.py` for the CLI/orchestration skeleton.
- **Deliverables:** `generate.py`; helper `pipeline.py` for the tokenizer→encode→PackedLayout→sample→decode→(video mux + audio mux) chain; simple `ffmpeg`-based muxer for combining decoded RGB frames (24 fps) + decoded 32 kHz stereo waveform into `.mp4`.
- **Effort:** ~4 days.
- **Risks:** ⚠ FFmpeg dependency (assume system install). ⚠ Nested-tensor packed sample scheduling — implement `NestedTensor` shim locally instead of pulling all of `comfy.nested_tensor`.

### Phase 8 — Quantization + benchmark (3–5 days)

- **Goal:** MLX 4-bit (`mx.quantize`, group_size=64) DiT + fp16 VAEs runs a 124-frame 768×1344 clip in reasonable wall-clock on M-series; produce a comparison table.
- **Deliverables:** `quantize.py` (in-place `mx.quantize` on Linear layers except the fp32 islands: `video_out`, `audio_out`, `video_patch_proj`, `audio_patch_proj`, `time_embedder.*`, `rope.inv_freq`). Benchmark script.
- **Effort:** ~4 days.
- **Risks:** ⚠ Quality degradation may need per-layer sensitivity sweep. ⚠ Peak RAM during sampling could still exceed 96 GB even after quantization — plan for on-disk streaming of layer-by-layer prefetch (like ComfyUI's `model_prefetch`).

## 7. Reference implementations checklist

| MLX file (this port) | ComfyUI reference | Line range (approx) |
|---|---|---|
| `attention.py` | `comfy/ldm/minimax/model.py :: Attention` | 141–174 |
| `rope.py` | `comfy/ldm/minimax/model.py :: rope_rotation_table`, `MiniMaxH3Model.rope_freqs` | 130–140, 508–518 |
| `blocks.py :: TimeEmbedder` | `comfy/ldm/minimax/model.py :: TimeEmbedder` | 114–128 |
| `blocks.py :: AdalnProj` | `comfy/ldm/minimax/model.py :: AdalnProj` | 187–201 |
| `blocks.py :: RefinerBlock` / `TokenRefiner` | same | 218–238 |
| `blocks.py :: DiTBlock` | same | 240–258 |
| `blocks.py :: FinalLayer` | same | 260–279 |
| `packed_layout.py` | `comfy/ldm/minimax/model.py :: PackedLayout` | 281–378 |
| `model.py :: MiniMaxH3Model` | same | 380–646 |
| `video_vae.py` | `comfy/ldm/minimax/vae.py` (whole file) | 1–694 |
| `audio_vae.py` | `comfy/ldm/minimax/audio_vae.py` (whole file) | 1–443 |
| `text_encoder_bridge.py` | `comfy/text_encoders/minimax.py` | 1–201 |
| `scheduler.py` | `comfy_extras/nodes_minimax_h3.py :: MiniMaxH3SigmaShift` + `comfy/model_sampling.ModelSamplingDiscreteFlow` | – |
| `generate.py` | `comfy_extras/nodes_minimax_h3.py` (whole) | 1–337 |
| `convert.py` | `comfy/sd.py :: VAE load detection` + our own `wan_2/convert.py` skeleton | – |

## 8. Known unknowns / risk register

- **MLX split-half RoPE fused kernel** — ComfyUI uses a custom CUDA/MPS kernel. Our first cut uses a plain functional implementation; expect a 1.5–2× DiT slowdown vs. ComfyUI on the same silicon until we add a Metal shim.
- **MLX `mx.fast.rms_norm` axis** — verify it operates on the last dim (it should); the H3 attention normalizes per-head so we may need to reshape before/after.
- **Text encoder wiring** — the Qwen3-VL-32B truncation is an unusual pattern. We have `~/models/Qwen3-VL-32B-Instruct-4bit` (MLX 4-bit) which is a *full* Qwen3-VL. Two options: (a) call the MLX Qwen but stop propagation at layer 50 (requires wrapping the model class), or (b) convert Ref2VA's own Qwen shards (already truncated). Option (b) is simpler but ~62 GB in bf16; option (a) reuses our existing 4-bit checkpoint. **Decision deferred to Phase 6.**
- **Vision-block preprocessing** — Qwen3-VL vision block for MiniMax uses `min_pixels=3136, max_pixels=12845056`, patch_size=16, temporal_patch_size=2, merge_size=2. Requires MLX bilinear resize (present) and a specific 8-way permute (`patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)`).
- **Reference audio VAE at encode time** — for `ref2va`, we must encode reference audio through our MLX audio VAE. This means the audio VAE encoder path is required, not just decode. Adds ~2 days to Phase 3.
- **NestedTensor** — ComfyUI wraps `(video, audio)` in a `NestedTensor` so any sampler can step both at once. We ship a 20-line local shim.
- **Attention op dispatch** — H3 uses `optimized_attention` (varlen packed). MLX's `mx.fast.scaled_dot_product_attention` is contiguous only. For the packed case we may need a masked sdpa or a manual softmax path. Prototype early.
- **`model_prefetch`** — ComfyUI ships a per-block layer-prefetch queue for streaming layers from disk / CPU on demand. We can skip this in Phase 4 and add later; on a 96 GB M3-Max the 29 GB bf16 DiT (or ~15 GB fp16, ~7 GB q4) should fit.
- **Missing curve-basis adaLN** — Ref2VA uses the plain time embedder. If MiniMax releases a curve variant we add the `use_adaln_curves` branch. Config field is already scaffolded.

## 9. Directory map

```
mlx_video/models/minimax_h3/
├── __init__.py               # exports
├── config.py                 # MiniMaxH3Config dataclass (implemented)
├── convert.py                # PyTorch/ComfyUI → MLX
├── generate.py               # CLI pipeline entry
├── model.py                  # MiniMaxH3Model (Phase 4)
├── blocks.py                 # DiTBlock, RefinerBlock, TimeEmbedder, AdalnProj, FinalLayer
├── attention.py              # H3 fused-RMSNorm + split-half rope attention
├── rope.py                   # 3D rope rotation table
├── packed_layout.py          # PackedLayout builder
├── video_vae.py              # Phase 2
├── audio_vae.py              # Phase 3
├── scheduler.py              # Phase 5
├── text_encoder_bridge.py    # Qwen3-VL layer-50 truncation wiring
├── pipeline.py               # tokenize → embed → sample → decode
├── PORT_PLAN.md              # this file
└── README.md                 # user-facing tracker
```

## 10. Phase 1 completion checklist

- [x] Clone ComfyUI + workflow templates
- [x] Grep-locate all H3 files
- [x] Deep-read `comfy/ldm/minimax/{model,vae,audio_vae}.py` (2321 LOC total across primary refs)
- [x] Deep-read `comfy/text_encoders/minimax.py` and `comfy_extras/nodes_minimax_h3.py`
- [x] Confirm weight name mapping (Ref2VA ↔ ComfyUI DiT: 1:1, no rename needed)
- [x] Confirm weight source choice (Ref2VA bf16 shards)
- [x] Study `mlx_video/models/wan_2/` for the port pattern to mirror
- [x] Scaffold `mlx_video/models/minimax_h3/` directory + stubs
- [x] Write `config.py` with real dataclass
- [x] Write `PORT_PLAN.md` (this doc)
- [x] Write `README.md`
- [x] Git branch `minimax-h3-port`, Phase 1 commit
- [x] Phase 2 kickoff

## 11. Phase 2 completion checklist (2026-08-04, one day vs 1-week estimate)

- [x] Read reference impls: ComfyUI vae.py (694 LOC) + Ref2VA/video_vae/*.py (12 files)
- [x] Write `VIDEO_VAE_NOTES.md` (architecture + shape flow + weight naming table)
- [x] Implement MLX `video_vae.py` (CausalConv3d, ResnetBlock3D, EncoderFCN3D,
      RotaryEmbeddingND, ViTAttention, TransformerBlock, ViT3DDecoder,
      MiniMaxH3VideoVAE) — 609 LOC
- [x] Implement `convert_video_vae()` in `convert.py` (Conv3d layout permute
      (O,I,D,H,W)→(O,D,H,W,I), bf16/fp16 output variants)
- [x] Convert Ref2VA source safetensors: 560 tensors, 2,603,871,032 params,
      35 conv3d transposed, bf16 output 4.97 GB in 17 s
- [x] Write `tests/test_h3_video_vae.py`: encoder parity, decoder parity,
      full round-trip on face image, benchmark
- [x] Verify all 4 tests pass — encoder 139 dB, decoder 63 dB, round-trip 63 dB
- [x] Record baseline benchmarks (5f/17f × 384×384)
- [x] Commit + update `PORT_PLAN.md` + `README.md`

### Phase 2 gotchas discovered

- **MLX has no `mx.flip`** in 0.31.2 — use negative-step slicing `x[..., ::-1, ...]`.
- **MLX `Conv3d` layout is `(O, D, H, W, I)`** (channel-last for both input and
  weights); the converter permutes PyTorch `(O, I, D, H, W)` on the way in.
- **`mx.pad` supports only `constant` and `edge`** — the reference uses
  `reflect` for spatial padding; we hand-roll reflect via slice-and-flip.
- **`use_t_isolated_gn=True`** in the config: GroupNorm3D reshapes 5-D input
  so that stats are computed per-frame. Our `TemporalIsolatedGroupNorm` does
  this directly on NDHWC to avoid a costly permute.
- **`qk_norm_affine=False`** → RMSNorm on q/k has *no* learnable weight and
  produces zero checkpoint keys. We keep only the functional path.
- **Model is designed for 17-frame clips**, not single images: the reference
  itself only round-trips a single image at ~15 dB (patch_size_t=4 means
  1 frame → 1 latent token → 4 output frames; the middle frames are
  interpolated). MLX matches the reference within 0.02 dB — the port is
  bit-faithful; the low absolute PSNR is intrinsic to the model on images.
- **Peak RSS is dominated by weights**: 5.0 GB bf16 static + ~0.3 GB peak
  activation for 17f × 384×384 = 5.3 GB total. Fits in any recent M-series.
- **Skipped for Phase 2 (deferred to Phase 4/7):** spatial tiling
  (`tile_size=256`), multi-clip temporal chunking with overlap-blend, and the
  `token_drop` cross-clip alignment. Fast path only.
<parameter name="timeout_ms">10000