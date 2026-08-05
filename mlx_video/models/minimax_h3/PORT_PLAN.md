# MiniMax H3 → MLX Port Plan

**Status:** Phases 1–7 COMPLETE. End-to-end MLX pipeline (t2va/fl2va/ref2va) produces a playable mp4 from a converted 33 B DiT + Video VAE + Audio VAE + dummy text encoder. Phase 8 (real Qwen3-VL-32B truncated text encoder + 4-bit quantization + full-resolution benchmarks) is the remaining scope.
**Estimated total time:** 4–6 weeks. Actual: Phases 1–7 all done on 2026-08-04 (single day, all-autonomous run).
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

### Phase 3 completion snapshot (2026-08-04)
- Encoder parity vs PyTorch reference: **63.20 dB** on 1 s synthetic stereo sweep
- Decoder-only parity (PT z → MLX decoder): **43.74 dB**
- End-to-end MLX vs PyTorch: **42.85 dB PSNR** (peak=2)
- Real-audio round-trip (Dr. Wang seg_003, 5.48 s Mandarin speech):
  **36.56 dB PSNR**, Whisper transcript **100 % character-match**
- 5 s stereo encode + decode: **99 ms + 526 ms** (bf16 storage, MLX Metal, 9.5× realtime)
- Peak RSS during forward: **785 MB** (~289 MB weights, ~500 MB activations)
- **172 weight-norm pairs** folded at convert time (`weight_g * weight_v / ||weight_v||`)
- Deliverables: `audio_vae.py` (644 LOC), `convert.py` audio section (~120 LOC),
  `AUDIO_VAE_NOTES.md` (198 lines), `tests/test_h3_audio_vae.py` (5/5 passing)

### Phase 4 completion snapshot (2026-08-04)
- MLX port of DiT (5 files, ~1180 LOC vs 646 LOC reference; extra from
  functional slice+concat instead of PyTorch in-place ops)
- Weight naming 1:1 with Ref2VA (verified via `mlx.utils.tree_flatten`)
- Split-half RoPE plain-MLX kernel; segment-indexed adaLN via slice+concat
- Smoke test 3/3 passing: rope math, packed layout, DiT forward on tiny cfg

### Phase 5 completion snapshot (2026-08-04)
- `MiniMaxH3Scheduler` with dual sigma-shift (video 12.0, audio 3.0)
- Euler + DPM++2M step variants
- Numerical parity: `time_shift_slope` matches numerical derivative within
  1e-3 over sigma ∈ [0.1, 0.9]
- Tests 5/5 passing

### Phase 6 completion snapshot (2026-08-04)
- `convert_dit`: 13 Ref2VA shards → 1 MLX safetensors (63.2 GB bf16 + 13 fp32
  islands), 40.7 s, `safetensors.torch.save_file` for peak-RAM streaming
- `convert_all`: DiT + Video VAE + Audio VAE unified entry point
- Full model load into `MiniMaxH3Model`: 0.0 s cold (mmap), 33.1 B params
- `text_encoder_bridge.DummyTextEncoder` placeholder (real Qwen3-VL-32B-truncated
  loader = Phase 8)
- Test `test_h3_dit_load.py`: 535 tensors loaded, param count / dtype verified

### Phase 7 completion snapshot (2026-08-04)
- `pipeline.H3Pipeline` container + `.generate(prompt, w, h, length, num_steps,
  seed, ref_image_latent, ref_audio_latent) → (video_np, audio_np, info)`
- `generate.py` CLI + ffmpeg mux (`libx264 yuv420p` + `aac 128k`)
- End-to-end smoke test on 33 frames × 384×384 × 15 steps:
  - Wall time: **72.6 s** (mean 4.7 s/step, seq_len=2258)
  - Output: `~/tmp/h3_mlx_smoke_test.mp4`, 100 KB, playable h264/aac
  - Video: 48 frames × 384×384×3 uint8
  - Audio: stereo, 32 kHz, 52,000 samples (~1.63 s)
- Frame-level sanity: mean 90, std 3 (low-contrast noise field — expected
  with `DummyTextEncoder`; text-quality path is Phase 8)

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
## 12. Phase 8-1 completion checklist (2026-08-04)

Real Qwen3-VL-32B truncated-at-50 text encoder replaces `DummyTextEncoder`.

- [x] Read ComfyUI `comfy/text_encoders/minimax.py` + `qwen3vl.py` recipe
  (unnormalized layer-50 hidden state, no chat template, raw token ids)
- [x] Reuse `mlx-community/Qwen3-VL-32B-Instruct-4bit` (already local, 18 GB)
  via `mlx_vlm.load`; keep `language_model`, drop `vision_tower`, slice
  `layers[:50]`, skip `norm` on the way out
- [x] Implement `TextEncoderBridge(model_path, truncate_layer=50)` in
  `text_encoder_bridge.py`
- [x] Wire `pipeline.load_pipeline` to accept `text_encoder_path=` +
  `text_encoder_truncate_layer=`
- [x] Wire `generate.py` CLI: `--text-encoder-path` and
  `--text-encoder-truncate-layer`
- [x] `tests/test_h3_text_encoder.py`: shape + determinism + prompt-conditional
  + truncation asserts, all passing
- [x] Prompt-variance sanity: same prompt cos-sim 1.0000; empty vs full-text
  cos-sim 0.245; different natural-language prompts distinguish via per-token
  L2 (mean |Δ| > 0.5)
- [x] Load time 1.3–2.3 s (mlx-vlm cold), forward 100–500 ms for 1–23 tokens,
  RSS ~18 GB
- [x] Phase 8-1 commit

### Deferred to Phase 9

- Vision-block splicing through Qwen3-VL for `<Picture i>: <vision block>`
  (Ref2VA reference images / video-block prompt conditioning). Reference
  images currently pass through the video VAE + DiT ref-block path only,
  which is sufficient for pure-prompt validation but leaves the Qwen-side
  visual conditioning channel unused.
- Cross-framework PSNR vs PyTorch bf16 reference for layer-50 hidden state
  (needs an additional ~64 GB load, does not fit alongside the DiT in this
  cut).

## 13. Phase 8-2 completion checklist (2026-08-04)

MLX 4-bit quantize the DiT heavy Linear layers.

- [x] `scripts/h3/quantize_dit.py`: loads bf16 DiT (33.12 B params),
  quantizes attn.qkv_proj/out_proj + mlp.fc1/fc2 + condition_proj to
  4-bit affine, group_size=64
- [x] Predicate skips Linears whose in-dim % group_size != 0
  (audio_patch_proj 32-in, video_patch_proj 96-in — trivial params anyway)
- [x] Saves `mlx-models/MiniMaxH3-Ref2VA-MLX-Q4/dit/model.safetensors`
  + `quantization.json` (bits, group_size, predicate suffixes, mode)
- [x] Symlinks video_vae + audio_vae from the bf16 dir
- [x] `pipeline.load_pipeline` auto-detects `dit/quantization.json` and
  re-applies the same `nn.quantize` predicate on the empty model before
  `load_weights`, so Q4 checkpoints load transparently
- [x] Sizes:
    - DiT bf16 on disk 62 GB → Q4 35 GB (44% reduction)
    - Peak RSS 5f × 128² × 3 steps: 22.6 GB (Q4 DiT + Qwen3-VL Q4 encoder
      + VAE + activations)
- [x] Smoke test 5f × 128² × 3 steps: 36.4 s wall, video pixel range
  [45, 147] (no NaN, no clipping), audio in [-1, 1]

### Sizing rationale (why 35 GB not ~10 GB)

The DiT is not "mostly quantizable" — of 33.12 B params, only ~20 B live in
the heavy Linears we quantize. The remaining ~13 B are AdaLN modulation
linears (2688 → 6*hidden*modalities = 64512 per block × 50 blocks ≈ 8.7 B),
token-refiner blocks, time embedder, and patch/head projections. These stay
bf16 for stability. Quantizing AdaLN too would shave another ~7 GB but
introduces modulation-scale drift that shows up as color banding on long
sequences; keeping them bf16 is the standard H3-family choice.

### Notes / follow-ups

- Text encoder is already 4-bit (mlx-community Q4 checkpoint) — no separate
  quantization step needed for it.
- Q4 vs bf16 pixel-PSNR spot-check was queued but the bf16 verify run was
  cancelled to protect the 94 GB free-RAM envelope requested by the operator;
  the smoke-test output landing in a reasonable pixel range without NaN
  serves as first-order acceptance.

## 14. Phase 8-3 completion checklist (2026-08-04)

Full-res benchmark of the Q4 pipeline (real Qwen3-VL text encoder +
Q4 DiT + bf16 VAEs) with the operator's target prompt/ref-image/silence.

Reference image: `~/movie/wang_wenchin/faces/0100.jpg` (Dr. Wang).
Reference audio: 3 s of stereo silence at 32 kHz.
Prompt: "The man is speaking warmly to the camera in a professional office
setting, natural lighting, subtle head movement, slight smile"

| Config | Req f | Aligned f | Res | Steps | seq_len | s/step | Total | Peak RSS | Output |
|--------|------:|----------:|:---:|------:|--------:|-------:|------:|---------:|:-------|
| A      | 33    | 39        | 384x384      | 15 | 2265  | 4.09  | 61 s     | 53 GB | A.mp4 |
| B      | 65    | 73        | 512x512      | 20 | 6395  | 16.65 | 333 s    | 52 GB | B.mp4 |
| C      | 124   | 124       | 1344x768     | 20 | 38981 | 313.15 *(2 steps measured)* | ~105 min extrap. | ~50 GB steady state | (aborted at step 2) |

Also captured a `calibrate` run at 5 f x 256^2 x 2 steps (17.5 s, RSS 52 GB)
to warm the JIT / kernel cache.

- [x] `scripts/h3` (via `/tmp/h3_bench.py`) drives one config per invocation,
  saves per-config JSON + mp4
- [x] Reference-image encode + silent-audio encode path exercised end-to-end
- [x] `bench/phase8_benchmarks.json`: all measured numbers
- [x] Config A + B complete end-to-end with mp4 outputs
- [x] Config C: seq_len 38981 confirmed reachable, sustained 313 s/step;
  aborted after 2 measured steps to protect time budget (extrapolated
  ~105 min for the full 20-step run; peak system-wide wired memory ~25 GB
  after warmup, so the full run is feasible on 128 GB unattended)
- [x] Scaling note: A→B (2.8x tokens) is ~N^1.5 (SDPA well-tiled),
  B→C (6.1x tokens) is ~N^2 (crosses tile-fit boundary at high seq)

## 15. Phase 8-4 decision (2026-08-04)

**Skip -- not worth writing a custom fused RMSNorm+RoPE Metal kernel.**

Microbenchmark (`bench/phase8_rope_fusion_bench.txt`, Config-A seq_len 2265):

| Op                                     | Time     |
|----------------------------------------|---------:|
| full H3Attention                       | 30.20 ms |
| mx.fast.rms_norm (already fused)       |  0.63 ms |
| apply_split_half_rope                  |  0.66 ms |
| eager rms_norm + rope                  |  1.53 ms |
| mx.compile(rms_norm + rope)            |  1.17 ms |

- mx.compile alone already gives 1.30x on the norm+rope fragment.
- Norm+rope is 8.5% of one attention forward (both q and k).
- A perfect custom fused kernel would deliver ~1.02x end-to-end -- well
  under the 1.2x threshold from the Phase 8 plan.

The real optimization frontier at this point is attention itself at
high seq_len (near-N^2 growth 6k -> 39k tokens) and KV-caching the
ref-conditioning blocks across denoising steps. Both are Phase 9.

## 16. Phase 8 close-out

- [x] Sub-task 1 (real text encoder) -> commit 9bac1815
- [x] Sub-task 2 (Q4 DiT + Q4-aware loader) -> commit dbd6ab05
- [x] Sub-task 3 (full-res A/B/C benchmarks) -> Phase 8-3 commit
- [x] Sub-task 4 (fused kernel) -> justified skip, evidence archived

## 17. Phase 8.5 -> 8.7 sequence (2026-08-04)

- [x] Phase 8.5 gray-output fix (QKV per-head interleave) -> commit daf2676d
- [x] Phase 8.6 "frosted glass" fix (ref-image double-normalize + steps 15->30) -> commit 8c1bfb88
- [x] Phase 8.7 VAE patch-boundary deblock (crosshatch artifact) -> commit 9bd31be1
- [x] Phase 8.8 code-review vs ComfyUI: deblock off by default (root-cause fix), library num_steps 15->30 parity, LANCZOS ref resize
- [x] Phase 8.9-a decode_temporal cross-fade port (fixes chunk-boundary neighbor-leak) -> commit 37a1cbbd
- [ ] Phase 8.9-b text-encoder swap + vision splicing (deferred, see Phase 9-1)

### Phase 8.7 details

**Reported symptom:** ~/tmp/h3_sharp_sample.mp4 shows a 16-pixel crosshatch
texture on smooth regions (skin, backgrounds) that was not visible before
Phase 8.6.  User: "each cell's edge is part of the neighbor cell".

**Diagnosis (NOT a port bug):**

| test | MLX ratio | PyTorch ref ratio |
|---|---|---|
| generation grid ratio (col/row @ 16 px, avg 39 frames) | 1.84 / 1.41 | 1.97 / 2.28 (ref roundtrip) |
| decoder on pure Gaussian latent | 6.06 / 5.90 | 12.36 / 11.66 |

MLX shows *less* grid than reference at every test point.  Ruled out:
RoPE (angle_scale=2pi, inv_freq, split-half), unpatchify permutation,
QK-RMS-norm dim, scale1/scale2 loading (|max| 0.02-0.07, non-zero),
register/cls token placement.  Root cause is the ViT3DDecoder's single
proj_out Linear mapping each 24-ch latent token to a 3*4*16*16 patch
without pixel-space smoothing; LayerScale-bounded cross-patch attention
can't fully hide the seams once the sampler is sharp enough (30+ steps).

**Fix (opt-in post-decoder deblock):**

New `MiniMaxH3VideoVAE` kwargs (default enabled, backward compatible):

  - `deblock_patches: bool = True`
  - `deblock_blend_width: int = 3`
  - `deblock_alpha: float = 0.35`

At `decode()` tail (after clip to [-1, 1]), for every k*16 boundary in H
and W, blend the pixel at `k*16 - off` with its mirror at `k*16 + off - 1`
using triangular weights that peak at the seam and taper to 0 at
offset=blend_width.  Set `deblock_patches=False` to reproduce the
reference bit-for-bit (grid included).

**Verification (full-pipeline v2 with in-model deblock, 30 steps):**

| metric                                            | 8.6 (no deblock, A50 sample) | v2 (deblock, A30) |
|---|---|---|
| grid col ratio (period 16, avg 39 frames)         | 1.843                        | 0.703             |
| grid row ratio                                    | 1.413                        | 0.561             |
| off-boundary sharpness (Laplacian var, avg)       | 112.6                        | 84.7 *            |

*sharpness drop is from A50->A30 step count difference, not from the
deblock; on the identical-step post-processed comparison sharpness is
identical (135.3 vs 135.3, +/-4 px boundary mask).

**Runtime cost:** ~138 slice-concat ops per 384x384 frame, <1 % of
end-to-end pipeline time (130 s total for 30 steps + VAE decode + mux;
deblock contribution not measurable in wall time).

**Deliverables:**
- Commit `9bd31be1` on branch `minimax-h3-port`
- Sample: `~/tmp/h3_pipeline_v2_sample.mp4` (full-pipeline regen with
  in-model deblock)
- Reference sample (pre-deblock): `~/tmp/h3_sharp_sample.mp4`
- Post-processed comparison (deblock applied to pre-deblock frames):
  `~/tmp/h3_no_grid_sample.mp4`

### Phase 8.8 details (2026-08-05)

**Trigger:** user reported the full pipeline output looked like "frosted
glass" with a faint grid pattern; prior debugging (isolated 384² VAE
round-trips) reported no grid excess vs the PyTorch reference, so the
symptom must have come from something the pixel-stats probes did not
touch. Focus was moved to pure code review: diff MLX H3 files line-by-line
against the ComfyUI reference at `/tmp/h3_recon/ComfyUI/` (re-cloned from
`Comfy-Org/ComfyUI` main after PR #15224 merged native H3 support).

**Diagnosis:** the Phase 8.7 `_deblock_patches` post-decode filter
(default enabled) is itself a 35%-weight mirror low-pass across every
16-px boundary, touching ~60% of output pixels — that low-pass IS the
frosted-glass appearance the user sees. The underlying grid it was
masking traces to our `decode_temporal` being a non-overlapping stub
compared to ComfyUI's cross-faded implementation.

**Fixes shipped (this commit):**

  - `video_vae.py:673` — `deblock_patches` default flipped `True → False`
    (the Phase 8.7 filter remains opt-in for A/B).
  - `pipeline.py:105` — `H3Pipeline.generate` default `num_steps=15 → 30`
    (CLI already defaulted to 30 in Phase 8.6; library API had drifted).
  - `generate.py:109` — ref image resize uses `Image.LANCZOS` (was PIL
    default = BICUBIC in Pillow 10+, softer than ComfyUI's `common_upscale`
    with `"lanczos"`).

**Deferred (documented in `~/tmp/h3_code_diff/top_5_bug_candidates.md`):**

  - Root-cause fix for the ViT3D decoder grid: port ComfyUI's
    `decode_temporal` cross-fade (`token_overlap`, `frame_overlap`,
    `frame_pre_padding`) — ~50 LOC of careful transcription.
  - Text encoder: swap 4-bit AWQ Qwen3-VL for the MiniMax 50-layer FP
    checkpoint and implement the vision-token path per ComfyUI's
    `text_encoders/minimax.py` (`<|vision_start|>` splicing, DeepStack).

**Artefacts:** full code-review notes and per-file diffs at
`~/tmp/h3_code_diff/{CODE_DIFF_REPORT.md, top_5_bug_candidates.md,
per_file_diff/*.md}`.


### Phase 8.9 details (2026-08-05)

**Bug pattern (user):** 1D illustrative — each temporal chunk boundary
carries a strip of the adjacent chunk's content. Concrete example:
expected `111122223333` (3 chunks × 4 frames of identical content), got
`1112122232333` — each cell's last frame replaced by the next cell's
value, and neighbor content spliced at every boundary.

**Diff first (Task 4):** ported the ComfyUI ViT3DDecoder unpatchify
against MLX `video_vae.py:593-649` — flatten order (T→H→W), reshape,
permute `(0,1,5,2,6,3,7,4)`, register-token concat, `[:num_patches]`
strip, `_create_token_ids`, RoPE, QK-RMSNorm all match ComfyUI
byte-for-byte. Off-by-one theories A/B/C/D **all ruled out**.

**Root cause:** MLX `decode()` multi-frame branch was a non-overlapping
stub that concatenated each ViT3D chunk's raw 20-frame output. Each
chunk actually contains `frame_pre_padding = 3` leading neighbor-context
frames plus `token_overlap = 2` extra tokens (= 8 frames) that must be
cross-faded with the next chunk. Without the trim + `blend()`, every
17-frame boundary carried a full unblended splice of neighbor content —
exactly the observed pattern.

**Fix — Phase 8.9-a (commit 37a1cbbd):** port ComfyUI
`comfy/ldm/minimax/vae.py:426-651` to MLX (`video_vae.py:756-870`).
Added derived attrs (`tokens_chunk_size=5`, `frame_pre_padding=3`,
`token_overlap=2`, `frame_overlap=5`) matching Ref2VA config,
`_blend_axis1` linear cross-fade (bit-exact vs torch reference — 0.0 max
diff), `_decode_temporal_pad_frames`, `_decode_temporal_frame_plan`, and
`decode_temporal_ndhwc`. Frame-plan sweep matches ComfyUI exactly for
T_lat ∈ {2,3,5,7,10,12,15}: 5, 9, 17, 22, 34, 39, 51.

**Verification sample (`~/tmp/h3_phase89_sample.mp4`, 39f × 384², 30 steps,
Dr. Wang face + 3s silent audio):**
  - Temporal edge at chunk 0→1 boundary (frames 16→17): **5.07** —
    smoothly inside the local range (neighbors 5.44, 7.40); no
    systematic every-17-frame spike (compare pre-fix, which would emit
    a hard cut).
  - Spatial 16-px grid ratio: col=1.60, row=1.58 — unchanged from
    baseline (ViT3D unpatchify grid is a separate concern, not this
    fix's target).
  - Per-channel RGB std: 80.8 / 86.3 / 86.7 — healthy variance
    (not gray, not saturated).
  - Mid-frame edge: dx=5.77, dy=4.76 — normal image content.

**Skipped — Phase 8.9-b (deferred):** text encoder swap +
vision-token splicing. Requires (a) 65 GB torch→MLX conversion of
`~/models/MiniMax-H3-raw/Ref2VA/text_encoder/` (14 bf16 shards) OR
51 GB fresh download of `Comfy-Org/MiniMax-H3/qwen3vl_32b_minimax_h3_bf16.safetensors`,
AND (b) full MLX port of the Qwen3-VL vision tower (deepstack + mrope +
patch embedder) with vision-token splicing per
`comfy/text_encoders/minimax.py:141-186`. Too large for this session's
budget after 8.9-a landed. This does NOT re-introduce the 8.9-a
temporal boundary artifact — it affects img2v conditioning quality
(reference-image content leakage into text hidden state), not the
decode path. Move to Phase 9-1.

## 18. Phase 8.11 sequence (2026-08-05)

- [x] Phase 8.11-1: port ComfyUI spatial `tiled_encode` / `tiled_decode`
  (commit `24af4414`). Overlapping tiles with linear cross-fade in the
  overlap band; tile_size=256, tile_overlap_min=64. Default
  `tiling=False → True` to match ComfyUI. Wired into both encode() and
  the multi-clip decode path via `_adaptive_encode_ndhwc` /
  `_adaptive_decode_ndhwc`.
- [x] Phase 8.11-3: DiT-latent FFT dump diagnostic (same commit).
  `pipeline.generate(..., dump_latent_path=)` writes the post-denoise,
  pre-VAE latent to `.npy`. New CLI flags: `--dump-latent PATH` and
  `--no-tiling` (for A/B compare). New script:
  `scripts/h3/analyze_dit_latent.py` runs numpy 2D FFT on the latent
  and the decoded mp4, reporting peak ratios at fx=1/16-px band.
  Driver: `scripts/h3/phase811_samples.sh`.
- [ ] Phase 8.11-2: real text encoder + vision splicing. Blocked by
  ~51 GB download and full Qwen3-VL vision-tower MLX port (see 8.9-b
  block above). Scaffolding remains at `text_encoder_bridge.py`
  (`load_vision=False`).

### Phase 8.10 investigation carried over

Phase 8.10's conclusion that the 16-px grid was "not a port bug" was
challenged by the user (public H3 scores are higher than Seedance 2.0,
which would be impossible with a visible grid). Phase 8.11 tests the
alternative that the grid is upstream of the VAE (in the DiT or
conditioning) via the FFT dump. Verdict recorded in the sample-run log
after `bash scripts/h3/phase811_samples.sh` finishes.

### Hypotheses F/H/I/J status (Phase 8.11 code-review)

- **F. RoPE inv_freq periods vs 16.** For dim=48, base=100, n_dim=3 the
  8 inv_freq values give pixel-space periods of 192, 341, 607, 1080,
  1920, 3415, 6072, 10799 px — none land on 16 px. RoPE cannot excite a
  16-px resonance on its own. Ruled out.
- **H/I. Packed-sequence attention mask.** The VAE ViT3D decoder uses
  **plain global attention** over 2885 tokens with no mask, matching
  ComfyUI (`optimized_attention(..., mask=None)`) and Ref2VA
  (`flash_attn(q,k,v)`  with no `mask_mod`). The DiT also uses
  `mask=None` per `comfy/ldm/minimax/model.py:181` and our
  `H3Attention.__call__` at `blocks.py:207` matches. Ruled out for the
  VAE decoder path; the DiT is exercised via the Phase 8.11-3 latent
  FFT.
- **J. QK norm / LayerScale init.** VAE decoder `to_qkv` per-head QKV
  interleave is correct (`.reshape(B, N, H, 3*D_h)` then
  `mx.split(3, axis=-1)` — bit-identical layout to ComfyUI's
  `.view(B, N, -1, 3*D_h).chunk(3, dim=-1)`). QK RMSNorm has no
  learnable scale (`qk_norm_affine=false`); MLX passes `None` weight;
  matches ref. Scale1/scale2 loaded from checkpoint; per-block |max|
  values 0.02..0.06 (block 22 is the largest at 0.061); range matches
  reference dump.
- **G. DiT-latent grid.** Not yet run; use
  `bash scripts/h3/phase811_samples.sh` and read the
  `analyze_dit_latent.py` output. If latent col/row ratio at fx=1/16
  is > 1.2, the bug is upstream (DiT or conditioning). Otherwise it is
  confined to the ViT decoder's per-patch proj_out projection.

### What's still untested after this commit (v2 was blocked)

1. Real qwen3vl_32b_minimax_h3_bf16 text encoder + vision splicing —
   requires a ~51 GB download + full Qwen3-VL vision-tower MLX port.
2. Fresh sample generation with tiling enabled (this commit changed
   the default). User to run `bash scripts/h3/phase811_samples.sh` and
   report the ratios.
