"""H3 pipeline glue: geometry helpers + full-model container + end-to-end
sample loop.

Port of the shape and encoding logic in ``comfy_extras/nodes_minimax_h3.py``.
We omit the multi-modal reference presentation for the Phase-7 smoke test —
the dummy text encoder replaces it — and keep only the plumbing needed to run
a t2va / fl2va / ref2va sample.

Public API
----------
    ``adapt_canvas(w, h) -> (canvas_w, canvas_h)``
    ``temporal_shape(length) -> (frame_count, video_latent_t, audio_latent_t)``

    ``H3Pipeline(dit, video_vae, audio_vae, text_encoder, scheduler)``
        .generate(prompt, width, height, length, ref_image=None,
                  ref_audio_wav=None, num_steps=15, seed=0) -> (video_np, audio_np)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .model import MiniMaxH3Model
from .packed_layout import PackedLayout, RefBlock
from .scheduler import MiniMaxH3Scheduler


# ---------------------------------------------------------------------------
# Geometry helpers (mirror the reference)
# ---------------------------------------------------------------------------

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
FPS = 24
AUDIO_LATENT_FPS = 40


def align_frame_count(n: int) -> int:
    """Snap ``n`` up to the next value ``n % 17 == 5`` (H3's frame grid)."""
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count: int) -> int:
    """Latent-time size for a given frame count (post align_frame_count)."""
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length: int) -> Tuple[int, int, int]:
    """Return (aligned_frame_count, video_latent_t, audio_latent_t)."""
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def adapt_canvas(width: int, height: int) -> Tuple[int, int]:
    """768-short-edge canvas with 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (
        max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class H3Pipeline:
    dit: Any  # MiniMaxH3Model
    video_vae: Any  # MiniMaxH3VideoVAE
    audio_vae: Any  # MiniMaxH3AudioVAE
    text_encoder: Any  # anything with .encode(prompt) -> [1, L, 5120] mx.array
    scheduler: MiniMaxH3Scheduler = field(default_factory=MiniMaxH3Scheduler)

    def build_adaln_cache_and_drop(
        self,
        num_steps: int,
        *,
        has_visual_cond: bool,
        has_audio_cond: bool,
        layout: Optional[Any] = None,
        model_hash: str = "unknown",
        lora_hash: Optional[str] = None,
        tag: str = "",
        verbose: bool = True,
    ):
        """Precompute per-block AdaLN modulation (step-indexed, v16) then drop weights.

        Rationale + design in ``modulation_cache.py``. Must be called AFTER any Turbo LoRA
        overlay (LoRA does not touch ``adaln_proj`` in this build, so this is only about
        the ordering of memory events) and BEFORE ``enable_layer_group_eviction`` so the
        eviction snapshot never captures the dropped adaln parameters.

        The optional ``layout`` argument lets the caller commit a specific
        :class:`PackedLayout` to the cache signature so it fails-closed when
        replayed against a differently-shaped run. Pass the same layout the
        upcoming ``generate()`` call will use.
        """
        import time as _time
        from .modulation_cache import (
            ModulationCache, DEFAULT_SCHEDULE_NAME, DropRefused,
            _live_signature_from_pipeline, drop_adaln_weights,
            hash_layout, per_step_unique_t, verify_cache_bitwise,
        )
        # Populate scheduler.sigmas so we know the union of unique-t values.
        self.scheduler.set_timesteps(num_steps)
        sigmas_list = self.scheduler.sigmas.tolist()
        step_ut = per_step_unique_t(
            sigmas_list,
            has_visual_cond=has_visual_cond,
            has_audio_cond=has_audio_cond,
            shift_video=self.scheduler.shift_video,
            shift_audio=self.scheduler.shift_audio,
        )
        if verbose:
            m_hist = [len(x) for x in step_ut]
            print(f"[adaln-cache] v16 step-indexed build: "
                  f"{len(step_ut)} steps M={m_hist} "
                  f"(visual_cond={has_visual_cond}, audio_cond={has_audio_cond})...")

        layout_hash = hash_layout(layout) if layout is not None else "unbound"
        signature = _live_signature_from_pipeline(
            num_steps=num_steps,
            num_blocks=len(self.dit.blocks),
            has_visual_cond=has_visual_cond,
            has_audio_cond=has_audio_cond,
            shift_video=self.scheduler.shift_video,
            shift_audio=self.scheduler.shift_audio,
            sigma_min=1e-5,
            schedule_name=DEFAULT_SCHEDULE_NAME,
            dtype="float32",
            use_adaln_curves=bool(getattr(self.dit, "use_adaln_curves", False)),
            cache_final=True,
            model_hash=model_hash,
            lora_hash=lora_hash,
            layout_hash=layout_hash,
            per_step_ut=step_ut,
            tag=tag,
        )
        t0 = _time.time()
        cache = ModulationCache.build(
            self.dit, step_ut, signature, dtype=mx.float32, cache_final=True,
        )
        # Attach BEFORE dropping so a botched drop still leaves an intact model behind a cache.
        self.dit._modulation_cache = cache
        build_s = _time.time() - t0
        if verbose:
            print(f"[adaln-cache] built in {build_s:.1f}s, "
                  f"size {cache.nbytes()/1024**2:.1f} MiB")

        # ---- v16 260807: bitwise verify then fail-closed drop. On any refusal the
        # cache stays attached and the ``adaln_proj`` weights are NOT dropped, so
        # the model still runs via its legacy path if the caller catches DropRefused.
        _n_checks, max_diff = verify_cache_bitwise(
            cache, self.dit,
            verify_final=cache.per_step_final is not None,
            verbose=verbose,
        )
        if max_diff != 0.0:
            # 30-step live-vs-cache diff != 0 -- log and refuse.
            print(f"[adaln-cache] REFUSING DROP: live-vs-cache max_diff={max_diff:.6e}")
            return cache

        t0 = _time.time()
        try:
            freed = drop_adaln_weights(
                self.dit, cache,
                live_signature=signature, drop_final=True,
                verify_diff=max_diff, verbose=verbose,
            )
        except DropRefused as exc:
            print(f"[adaln-cache] {exc} -- keeping legacy adaln_proj weights, "
                  f"cache still attached")
            return cache
        drop_s = _time.time() - t0
        if verbose:
            print(f"[adaln-cache] dropped adaln_proj weights, "
                  f"freed {freed/1024**3:.2f} GiB in {drop_s:.1f}s")
        # Sanity assertion -- dropping the 50-block adaln bank should free >=20 GiB
        # (bf16 in mlx-video: ~26 GiB; Q4/Q8: less but still large). If well under, the
        # drop path missed arrays and we should not silently proceed.
        assert freed >= 20 * 1024**3, (
            f"[adaln-cache] freed only {freed/1024**3:.2f} GiB, expected >=20 GiB "
            "-- drop path likely missed base/LoRA arrays"
        )
        return cache

    def reset_adaln_cache(self):
        """Detach any attached ModulationCache. Call at ``generate()`` start when
        the schedule / conditioning may have changed between runs."""
        if getattr(self.dit, "_modulation_cache", None) is not None:
            self.dit._modulation_cache = None

    def enable_layer_group_eviction(self, group_size: int = 10, verbose: bool = True):
        """Install a LayerGroupManager on ``self.dit``.

        Splits the 50 DiT blocks into groups of ``group_size`` and evicts
        each group's weights to CPU-side numpy arrays. During forward,
        one group is re-activated at a time; other groups are dormant so
        their weights do not count against Metal's wired budget.

        Call this AFTER any Turbo LoRA overlay has been installed so LoRA
        A/B tensors are captured in the dormant state.
        """
        from .layer_group_evict import LayerGroupManager
        mgr = LayerGroupManager(self.dit, group_size=group_size, verbose=verbose)
        self.dit._layer_mgr = mgr
        return mgr

    def _empty_av_latents(self, width: int, height: int, frame_count: int, dtype: mx.Dtype = mx.float32):
        _, latent_t, audio_t = temporal_shape(frame_count)
        video = mx.zeros((1, 24, latent_t, height // 16, width // 16), dtype=dtype)
        audio = mx.zeros((1, 32, 2, audio_t), dtype=dtype)
        return video, audio, latent_t, audio_t

    def generate(
        self,
        prompt: str = "",
        width: int = 384,
        height: int = 384,
        length: int = 5,
        # Phase 8.8: default raised 15 -> 30 to match the CLI. At 15 steps the
        # final Euler leg integrates a sigma delta of ~-0.46 in a single shot,
        # which visibly softens output. CLI generate.py already defaulted to 30
        # (Phase 8.6) but the library API had drifted.
        num_steps: int = 30,
        seed: int = 0,
        ref_image_latent: Optional[mx.array] = None,
        ref_video_latent: Optional[mx.array] = None,
        ref_audio_latent: Optional[mx.array] = None,
        verbose: bool = True,
        # Phase 8.11-3 diagnostic: dump the DiT-produced latent (post-denoise,
        # pre-VAE-decode) to this path as an npy file. Use with the
        # scripts/h3/analyze_dit_latent.py FFT tool to check whether the
        # 16-px spatial grid is already present upstream of the VAE.
        dump_latent_path: Optional[str] = None,
        # Phase 8.9-b: pre-computed text conditioning. If provided, the
        # attached ``text_encoder`` is NOT called — this lets the caller run
        # ``H3TextEncoderBridge.encode(...)`` in a separate stage, free the
        # 51 GB encoder, then load the DiT (RAM plan B, since encoder + DiT
        # together exceed the 128 GB budget on this machine).
        context: Optional[mx.array] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """End-to-end t2va (with optional ref image / ref audio).

        Returns
        -------
        video_np : uint8 [T_frames, H, W, 3]
        audio_np : float32 [T_samples] (mono) — pipe both channels concatenated
        info     : {'wall_time_s', 'peak_rss_gb', 'num_steps', ...}
        """
        import time as _time
        import resource as _resource

        t0 = _time.time()
        frame_count, latent_t, audio_t = temporal_shape(length)
        if verbose:
            print(f"[H3] frame_count={frame_count}, latent_t={latent_t}, audio_t={audio_t}")

        # ------- 1) Initial noise (video + audio) -------
        rng = np.random.default_rng(seed)
        video_latent = mx.array(rng.standard_normal((1, 24, latent_t, height // 16, width // 16)).astype(np.float32))
        audio_latent = mx.array(rng.standard_normal((1, 32, 2, audio_t)).astype(np.float32))

        # ------- 2) Text embeddings (fp32) -------
        # Phase 8.9-b: allow a pre-computed context to bypass the attached
        # encoder (see RAM plan B in the ``context`` docstring above).
        if context is None:
            context = self.text_encoder.encode(prompt).astype(mx.float32)
        else:
            context = context.astype(mx.float32)
        text_len = context.shape[1]

        # ------- 3) Set up scheduler + noise scale -------
        self.scheduler.set_timesteps(num_steps)
        video_latent = self.scheduler.scale_noise(video_latent)
        audio_latent = self.scheduler.scale_noise(audio_latent)

        # ------- 4) Build ref blocks (optional, ref2va path) -------
        refs: List[RefBlock] = []
        payload: Dict[str, Any] = {"seed": seed}
        cond_video_latents = []
        cond_audio_latents = []
        if ref_image_latent is not None:
            # ref_image_latent shape: [1, 24, 1, h, w]
            _, _, _, rh, rw = ref_image_latent.shape
            refs.append(RefBlock(kind="image", latent_h=rh, latent_w=rw))
            cond_video_latents.append(ref_image_latent)
        if ref_video_latent is not None:
            # ref_video_latent shape: [1, 24, vt, h, w]
            _, _, vt, rh, rw = ref_video_latent.shape
            refs.append(RefBlock(kind="video", latent_h=rh, latent_w=rw, latent_t=vt))
            cond_video_latents.append(ref_video_latent)
        if ref_audio_latent is not None:
            # ref_audio_latent shape: [1, 32, 2, T]
            rat = ref_audio_latent.shape[-1]
            refs.append(RefBlock(kind="audio", ref_audio_t=rat))
            cond_audio_latents.append(ref_audio_latent)
        if refs:
            payload["refs"] = refs
            payload["cond_video_latents"] = cond_video_latents
            payload["cond_audio_latents"] = cond_audio_latents

        # Cache layout across steps
        layout = PackedLayout(
            text_len, latent_t, height // 16, width // 16, audio_t,
            refs=refs if refs else None,
        )
        payload["layout"] = layout
        if verbose:
            print(f"[H3] packed seq_len={layout.seq_len}")

        # v16 260807: fail-closed if an attached ModulationCache disagrees with the
        # run about num_steps / conditioning / schedule. Refuse to launch rather
        # than silently produce a garbage output.
        cache = getattr(self.dit, "_modulation_cache", None)
        if cache is not None:
            sig = cache.signature
            problems = []
            if sig.num_steps != num_steps:
                problems.append(f"num_steps mismatch: cache={sig.num_steps} run={num_steps}")
            has_vis = bool(cond_video_latents)
            has_aud = bool(cond_audio_latents)
            if sig.has_visual_cond != has_vis:
                problems.append(f"has_visual_cond mismatch: cache={sig.has_visual_cond} run={has_vis}")
            if sig.has_audio_cond != has_aud:
                problems.append(f"has_audio_cond mismatch: cache={sig.has_audio_cond} run={has_aud}")
            if sig.num_blocks != len(self.dit.blocks):
                problems.append(f"num_blocks mismatch: cache={sig.num_blocks} run={len(self.dit.blocks)}")
            if problems:
                raise RuntimeError(
                    "[adaln-cache] cache signature disagrees with run; refusing: "
                    + "; ".join(problems)
                    + " -- call pipe.reset_adaln_cache() and rebuild for this run"
                )

        # ------- 5) Denoise loop -------
        for i in range(num_steps):
            step_t0 = _time.time()
            # v16 260807: step-indexed cache lookup. The DiT reads this to index the
            # ModulationCache; it also serves as a canonical run-position marker.
            payload["step_index"] = i
            ts = self.scheduler.timestep_for(i)
            v_video, v_audio = self.dit(
                (video_latent, audio_latent), ts, context, payload=payload,
            )
            mx.eval(v_video, v_audio)
            video_latent, audio_latent = self.scheduler.step(
                v_video, v_audio, i, video_latent, audio_latent,
            )
            mx.eval(video_latent, audio_latent)
            if verbose:
                sigma = float(self.scheduler.sigmas[i])
                print(f"[H3] step {i+1}/{num_steps}: sigma={sigma:.4f}, "
                      f"step={_time.time()-step_t0:.1f}s")

        # ------- 5.5) Optional: dump DiT-produced latent for FFT diagnostics -------
        if dump_latent_path is not None:
            from pathlib import Path as _Path
            _p = _Path(dump_latent_path).expanduser()
            _p.parent.mkdir(parents=True, exist_ok=True)
            lat_np = np.asarray(video_latent.astype(mx.float32))
            np.save(str(_p), lat_np)
            if verbose:
                print(f"[H3] dumped DiT latent to {_p}  (shape={lat_np.shape})")

        # ------- 6) Decode video + audio via VAEs -------
        if verbose:
            print("[H3] decoding video...")
        video_pixels = self.video_vae.decode(video_latent)  # [1, C=3, T, H, W]
        mx.eval(video_pixels)
        if verbose:
            print("[H3] decoding audio...")
        audio_waveform = self.audio_vae.decode(audio_latent)  # [1, C=2, T]
        mx.eval(audio_waveform)

        # ------- 7) Convert to numpy -------
        video_np = np.asarray(video_pixels).astype(np.float32)
        video_np = np.clip((video_np + 1.0) * 0.5, 0.0, 1.0)  # [-1,1] -> [0,1]
        video_np = (video_np * 255).astype(np.uint8)
        # video_np: [1, C=3, T, H, W] -> [T, H, W, 3]
        video_np = video_np[0].transpose(1, 2, 3, 0)

        audio_np = np.asarray(audio_waveform).astype(np.float32)[0]  # [C=2, T]

        # v15 260807 bugfix #4: on macOS ru_maxrss is BYTES (not KiB as on Linux).
        # bytes / (1024**3) -> GiB. Previous code did bytes / 1024 / 1024 -> MiB
        # but labelled it "GB" (produced values like 25779 "GB" that were actually MiB).
        import sys as _sys
        _ru = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        if _sys.platform == "darwin":
            process_rss_gib = _ru / 1024**3          # bytes -> GiB
        else:
            process_rss_gib = _ru / 1024 / 1024      # KiB   -> GiB
        info = {
            "wall_time_s": _time.time() - t0,
            "process_rss_gib": process_rss_gib,
            # Back-compat alias (deprecated): older callers read this key.
            "peak_rss_gb": process_rss_gib,
            "num_steps": num_steps,
            "frame_count": frame_count,
            "video_shape": tuple(video_np.shape),
            "audio_shape": tuple(audio_np.shape),
            "seq_len": layout.seq_len,
        }
        return video_np, audio_np, info


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def load_pipeline(
    model_root: Path = Path("~/mlx-video/mlx-models/MiniMaxH3-Ref2VA-MLX-bf16"),
    text_encoder=None,
    text_encoder_path: Optional[str] = None,
    text_encoder_truncate_layer: int = 50,
) -> H3Pipeline:
    """Load a fully-assembled H3 pipeline from a converted model directory.

    Text encoder resolution (in order):
      1. ``text_encoder`` (any object with ``.encode(prompt) -> mx.array``)
      2. ``text_encoder_path`` -> build ``TextEncoderBridge`` (Phase 8-1)
      3. Fallback: ``DummyTextEncoder`` (Phase-7 smoke test)
    """
    from .config import MiniMaxH3Config
    from .video_vae import MiniMaxH3VideoVAE
    from .audio_vae import MiniMaxH3AudioVAE
    from .text_encoder_bridge import DummyTextEncoder, TextEncoderBridge

    model_root = Path(model_root).expanduser()

    cfg = MiniMaxH3Config()
    dit = MiniMaxH3Model(cfg)

    # Detect Q4 checkpoint and re-quantize the empty model to match layout
    dit_dir = model_root / "dit"
    qmeta = dit_dir / "quantization.json"
    if qmeta.exists():
        import json as _json
        import mlx.nn as _nn
        meta = _json.loads(qmeta.read_text())
        suffixes = tuple(meta["class_predicate_suffixes"])
        gs = int(meta["group_size"])
        bits = int(meta["bits"])
        def _pred(path, m):
            if not hasattr(m, "to_quantized"):
                return False
            if not any(path.endswith(s) for s in suffixes):
                return False
            w = getattr(m, "weight", None)
            if w is not None and w.ndim >= 2 and w.shape[-1] % gs != 0:
                return False
            return True
        _nn.quantize(dit, group_size=gs, bits=bits, class_predicate=_pred)

    dit.load_weights(str(dit_dir / "model.safetensors"))

    video_vae = MiniMaxH3VideoVAE()
    video_vae.load_weights(str(model_root / "video_vae" / "model.safetensors"), strict=False)

    audio_vae = MiniMaxH3AudioVAE()
    audio_vae.load_weights(str(model_root / "audio_vae" / "model.safetensors"), strict=False)

    if text_encoder is None:
        if text_encoder_path is not None:
            text_encoder = TextEncoderBridge(text_encoder_path,
                                             truncate_layer=text_encoder_truncate_layer)
        else:
            text_encoder = DummyTextEncoder()

    scheduler = MiniMaxH3Scheduler()
    return H3Pipeline(dit=dit, video_vae=video_vae, audio_vae=audio_vae,
                      text_encoder=text_encoder, scheduler=scheduler)
