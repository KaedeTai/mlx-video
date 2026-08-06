"""Phase 8.9-b end-to-end sample using RAM plan B.

1) Load H3TextEncoderBridge alone -> encode prompt -> free encoder
2) Load H3 pipeline (DiT + video VAE + audio VAE) with DummyTextEncoder
3) Call pipe.generate(..., context=<pre-computed>)
4) Mux to mp4

This avoids the 51 GB encoder and ~30-60 GB DiT/VAE stack ever coexisting
in RAM (max 128 GB on this box).

Usage:
    python -m scripts.h3.sample_phase89b \
        --prompt "王博士溫暖地看著鏡頭說：大家好，我是王文欽博士" \
        --ref-image ~/tmp/wang_0100.jpg \
        --width 384 --height 576 --length 33 --num-steps 30 \
        --output ~/tmp/h3_phase89b_v1_sample.mp4
"""

from __future__ import annotations

import argparse
import gc
import resource as _resource
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


def _log(msg: str):
    print(f"[phase89b {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _peak_rss_gib() -> float:
    """Process peak RSS in GiB.

    v15 260807 bugfix #4: on macOS ``ru_maxrss`` is BYTES, on Linux it is KiB.
    Convert to GiB accordingly. Historically this function divided bytes by
    1024**3 (correct on darwin) but the callers labelled the number "GB" while
    also using a helper name ``_peak_rss_gb`` -- keep the label as ``GiB``
    everywhere to match the arithmetic and avoid the 25779 "GB" -> 25.17 GiB
    mislabel bug.

    v15 260807 bugfix #5: process RSS on macOS does NOT include Metal
    driver-wired memory that MLX allocates via the IOKit path -- the true
    footprint can be 50-66 GiB while ``ru_maxrss`` reports only 25 GiB. Use
    ``_system_metal_footprint_gib()`` (vm_stat wired+active pages) to see
    what the OS actually thinks the process family is using.
    """
    import sys as _sys
    ru = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    if _sys.platform == "darwin":
        return ru / 1024**3   # bytes -> GiB
    return ru / 1024 / 1024   # KiB -> GiB


# Cache the page size so we don't shell out for every log line.
_VM_PAGE_SIZE_CACHE: int = 0


def _vm_page_size() -> int:
    global _VM_PAGE_SIZE_CACHE
    if _VM_PAGE_SIZE_CACHE:
        return _VM_PAGE_SIZE_CACHE
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            if "page size of" in line:
                # e.g. "Mach Virtual Memory Statistics: (page size of 16384 bytes)"
                _VM_PAGE_SIZE_CACHE = int(line.split("page size of")[1].split("bytes")[0].strip())
                return _VM_PAGE_SIZE_CACHE
    except Exception:
        pass
    _VM_PAGE_SIZE_CACHE = 16384  # apple silicon default
    return _VM_PAGE_SIZE_CACHE


def _system_metal_footprint_gib() -> float:
    """System-wide (wired + active) memory in GiB from ``vm_stat``.

    On macOS the DiT weights that MLX pushes to the GPU end up in Metal-wired
    or Metal-active pages that DO NOT count against per-process RSS. To see
    the real "how much RAM is this run consuming" number we need the OS-level
    wired+active total. Not perfectly attributable to this Python process --
    other processes contribute too -- but delta between snapshots taken
    before/after model load is a reliable Metal footprint proxy.
    """
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    except Exception:
        return 0.0
    wired_pages = active_pages = 0
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Pages wired down:"):
            wired_pages = int(line.rsplit(":", 1)[1].strip().rstrip("."))
        elif line.startswith("Pages active:"):
            active_pages = int(line.rsplit(":", 1)[1].strip().rstrip("."))
    return (wired_pages + active_pages) * _vm_page_size() / 1024**3


def _mem_snapshot() -> str:
    """Compact one-line memory snapshot: process RSS + system wired+active."""
    try:
        import mlx.core as _mx
        mlx_active = _mx.get_active_memory() / 1024**3
        mlx_peak = _mx.get_peak_memory() / 1024**3
        mlx_str = f", mlx_active={mlx_active:.2f} GiB, mlx_peak={mlx_peak:.2f} GiB"
    except Exception:
        mlx_str = ""
    return (f"process_rss={_peak_rss_gib():.2f} GiB, "
            f"system_metal_footprint={_system_metal_footprint_gib():.2f} GiB"
            f"{mlx_str}")


# Back-compat alias so any older log line still runs.
_peak_rss_gb = _peak_rss_gib


def _run_ffmpeg(video_rgb: np.ndarray, audio_stereo: np.ndarray,
                fps: int, sample_rate: int, out_path: Path):
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, C = video_rgb.shape
    assert C == 3 and audio_stereo.shape[0] == 2
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_video = td / "video.raw"
        raw_audio = td / "audio.wav"
        raw_video.write_bytes(video_rgb.tobytes())
        audio_i = audio_stereo.T.astype(np.float32)
        audio_int16 = (np.clip(audio_i, -1.0, 1.0) * 32767).astype(np.int16)
        try:
            from scipy.io import wavfile
            wavfile.write(str(raw_audio), sample_rate, audio_int16)
        except ImportError:
            import wave
            with wave.open(str(raw_audio), "wb") as wf:
                wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
        # v15 260807 bugfix #6: previously used "-shortest" which trims to whichever
        # stream ends first. Because H3's audio VAE occasionally emits a waveform that
        # is a few samples shorter than exactly ``T / fps`` seconds, "-shortest" was
        # dropping the last frame (56 -> 55 in the observed case). Fix: explicitly
        # tell ffmpeg the exact video frame count and drop -shortest. The video
        # stream is authoritative; ffmpeg will pad audio with silence if needed.
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(fps), "-i", str(raw_video),
            "-i", str(raw_audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
            "-frames:v", str(T),
            "-c:a", "aac", "-b:a", "128k",
            # Pad audio with silence to at least video length so audio isn't cut short.
            "-af", "apad",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Post-mux frame-count sanity check: if the resulting mp4 has fewer video
        # frames than requested, surface the error immediately.
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-count_frames", "-show_entries", "stream=nb_read_frames",
                 "-of", "default=nk=1:nw=1", str(out_path)],
                capture_output=True, text=True, check=True,
            )
            got = int(probe.stdout.strip())
            if got != T:
                raise RuntimeError(
                    f"[mux] frame count mismatch: wrote {got}, expected {T} "
                    f"(video_rgb has {T} frames)"
                )
        except (subprocess.CalledProcessError, ValueError) as e:
            # Non-fatal: probe unavailable / unparseable -- still emit a warning
            print(f"[mux] warning: could not verify frame count: {e!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    # v17 multiref 260807: --ref-image / --ref-video / --ref-audio are
    # repeatable. Pass the flag multiple times to add more refs, in Comfy
    # standard order (images -> videos -> audios). Back-compat: passing
    # exactly one still works identically to the old single-ref API.
    ap.add_argument("--ref-image", default=None, action="append",
                    help="Reference image (repeatable, up to 9 per Comfy).")
    ap.add_argument("--ref-video", default=None, action="append",
                    help="Reference video mp4 (repeatable, up to 3 per Comfy). "
                         "Encoded via video_vae as kind='video' RefBlock (audio track "
                         "is currently dropped; paired video_audio is Phase C sub 5).")
    ap.add_argument("--ref-audio", default=None, action="append",
                    help="Reference audio (repeatable, up to 3 per Comfy).")
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--length", type=int, default=33)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder-path",
                    default="~/mlx-video/mlx-models/H3-TextEncoder-MLX-Q4")
    ap.add_argument("--model-root",
                    default="~/mlx-video/mlx-models/MiniMaxH3-Ref2VA-MLX-Q4")
    ap.add_argument("--output", required=True)
    ap.add_argument("--context-npy", default=None,
                    help="Optional: skip encode step, load a pre-saved context")
    ap.add_argument("--save-context", default=None,
                    help="Also save the encoded context to this .npy path")
    ap.add_argument("--dry-encode", action="store_true",
                    help="Just encode and save context; skip DiT")
    ap.add_argument("--turbo-lora", default=None,
                    help="Path to Turbo LoRA safetensors for 4-step sampling (runtime overlay)")
    ap.add_argument("--turbo-lora-alpha", default=1.0, type=float,
                    help="LoRA scale factor (default 1.0)")
    ap.add_argument("--adaln-cache", action="store_true",
                    help="v15: precompute per-block AdaLN modulation from bf16 adaln_proj, "
                         "then drop those weights. Cuts DiT resident ~24 GB with no voice-quality "
                         "loss (cache built from bf16, not Q4).")
    ap.add_argument("--layer-group-size", type=int, default=0,
                    help="If >0, enable LayerGroupManager on the DiT blocks. "
                         "Weights of dormant groups live in CPU RAM (not Metal-wired), "
                         "trading ~+30-40 ms/step for a lower peak Metal working set.")
    ap.add_argument("--phase-evict", action="store_true",
                    help="v16 Sub4 phase-aware residency: evict video_vae after refs "
                         "have been encoded (before denoise) and reload it before "
                         "decode. Also evicts the DiT before decode. Trades ~5-10 s "
                         "of extra VAE reload wall time for a lower peak Metal-wired "
                         "working set. Requires loading via load_pipeline() so the "
                         "model_root is known.")
    ap.add_argument("--eval-every", type=int, default=10,
                    help="v16 Sub3 materialisation barrier: mx.eval(h) every N DiT "
                         "blocks in the forward loop. 0 disables. Default 10. Lets "
                         "the arena drop older activations mid-step to cap peak Metal. "
                         "Ignored when --layer-group-size > 0 (group-eviction already "
                         "materialises at group boundaries).")
    args = ap.parse_args()

    _log(f"mem(start): {_mem_snapshot()}")

    # v17 multiref: normalise to lists of resolved paths.
    ref_image_paths = [Path(x).expanduser() for x in (args.ref_image or [])]
    ref_video_paths = [Path(x).expanduser() for x in (args.ref_video or [])]
    ref_audio_paths = [Path(x).expanduser() for x in (args.ref_audio or [])]
    # Back-compat singletons (None if empty, first element otherwise).
    ref_image_path = ref_image_paths[0] if ref_image_paths else None
    ref_video_path = ref_video_paths[0] if ref_video_paths else None
    ref_audio_path = ref_audio_paths[0] if ref_audio_paths else None
    _log(f"multiref: {len(ref_image_paths)} image(s), "
         f"{len(ref_video_paths)} video(s), {len(ref_audio_paths)} audio(s)")

    # ---------- Stage 1: text encoder ----------
    if args.context_npy is None:
        from mlx_video.models.minimax_h3.text_encoder_bridge import H3TextEncoderBridge
        import mlx.core as mx

        _log("stage 1: loading H3TextEncoderBridge...")
        t0 = time.time()
        enc = H3TextEncoderBridge(model_path=args.encoder_path)
        _log(f"encoder loaded in {time.time()-t0:.1f}s, {_mem_snapshot()}")

        _log(f"encoding prompt: {args.prompt!r}")
        _log(f"  n_ref_image={len(ref_image_paths)}, "
             f"n_ref_video={len(ref_video_paths)}, "
             f"n_ref_audio={len(ref_audio_paths)}")
        t0 = time.time()
        ctx = enc.encode(args.prompt,
                         has_ref_image=len(ref_image_paths),
                         has_ref_video=len(ref_video_paths),
                         has_ref_audio=len(ref_audio_paths))
        mx.eval(ctx)
        _log(f"encode done in {time.time()-t0:.2f}s, "
             f"context shape={ctx.shape}, dtype={ctx.dtype}, "
             f"norm={float(mx.linalg.norm(ctx)):.3f}, "
             f"mean={float(mx.mean(ctx)):.4f}, "
             f"std={float(mx.std(ctx)):.4f}")

        ctx_np = np.asarray(ctx)
        if args.save_context:
            save_p = Path(args.save_context).expanduser()
            save_p.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_p, ctx_np)
            _log(f"saved context to {save_p}")

        # Free the encoder ---------------------------------------------
        del enc, ctx
        gc.collect()
        try:
            mx.clear_cache()  # release wired arenas
        except AttributeError:
            pass
        _log(f"encoder freed, {_mem_snapshot()}")
    else:
        _log(f"loading pre-saved context from {args.context_npy}")
        ctx_np = np.load(Path(args.context_npy).expanduser())
        _log(f"context shape={ctx_np.shape}")

    if args.dry_encode:
        _log("dry-encode: exiting before DiT load")
        return

    # ---------- Stage 2: DiT + VAEs ----------
    _log("stage 2: loading H3 pipeline (DiT + VAEs) with DummyTextEncoder...")
    t0 = time.time()
    from mlx_video.models.minimax_h3.pipeline import load_pipeline
    pipe = load_pipeline(Path(args.model_root).expanduser())
    _log(f"pipeline loaded in {time.time()-t0:.1f}s, {_mem_snapshot()}")

    if args.turbo_lora:
        from mlx_video.models.minimax_h3.lora import load_turbo_lora
        _log(f"loading Turbo LoRA: {args.turbo_lora} (alpha={args.turbo_lora_alpha})")
        t_l = time.time()
        n_wrapped = load_turbo_lora(pipe.dit, args.turbo_lora,
                                     alpha=args.turbo_lora_alpha, verbose=True)
        _log(f"Turbo LoRA installed: {n_wrapped} modules in {time.time()-t_l:.1f}s, {_mem_snapshot()}")

    # v15: AdaLN cache before LayerGroupManager so eviction snapshot excludes dropped adaln.
    # v16 Sub2: auto-detect stripped bundle -- if <model_root>/dit/modulation_cache.npz
    # exists we load the cache directly instead of rebuilding (and the adaln_proj weights
    # were never in the model to begin with, so no drop is needed).
    dit_dir = Path(args.model_root).expanduser() / "dit"
    stripped_cache_npz = dit_dir / "modulation_cache.npz"
    stripped_cache_sig = dit_dir / "cache_signature.json"
    is_stripped_bundle = stripped_cache_npz.exists() and stripped_cache_sig.exists()

    if args.adaln_cache and is_stripped_bundle:
        _log(f"v16 Sub2: detected stripped bundle at {dit_dir} -- loading cache from disk")
        t_c = time.time()
        import mlx.core as _mx_c
        active_before = _mx_c.get_active_memory() / 1024**3
        from mlx_video.models.minimax_h3.modulation_cache import load_cache_bundle
        cache = load_cache_bundle(stripped_cache_npz, stripped_cache_sig)
        pipe.dit._modulation_cache = cache
        # Fail-closed signature check for the loaded bundle vs. the run config.
        sig = cache.signature
        problems = []
        if sig.num_steps != args.num_steps:
            problems.append(f"num_steps: bundle={sig.num_steps} run={args.num_steps}")
        has_vis = bool(ref_image_paths or ref_video_paths)
        has_aud = bool(ref_audio_paths)
        if sig.has_visual_cond != has_vis:
            problems.append(f"has_visual_cond: bundle={sig.has_visual_cond} run={has_vis}")
        if sig.has_audio_cond != has_aud:
            problems.append(f"has_audio_cond: bundle={sig.has_audio_cond} run={has_aud}")
        if sig.num_blocks != len(pipe.dit.blocks):
            problems.append(f"num_blocks: bundle={sig.num_blocks} run={len(pipe.dit.blocks)}")
        if problems:
            raise SystemExit(
                "[stripped-bundle] signature mismatch -- refusing to run: "
                + "; ".join(problems)
                + f"\nRebuild the bundle with matching options or point --model-root "
                  f"at the full (non-stripped) bundle."
            )
        active_after = _mx_c.get_active_memory() / 1024**3
        _log(f"stripped bundle cache loaded: {cache.num_steps} steps, "
             f"{cache.nbytes()/1e6:.1f} MB lookup table, "
             f"tag={sig.tag!r}, lora_hash={sig.lora_hash}")
        _log(f"active_mlx {active_before:.2f} GiB -> {active_after:.2f} GiB "
             f"in {time.time()-t_c:.1f}s, {_mem_snapshot()}")
    elif args.adaln_cache:
        _log("v16: building AdaLN modulation cache and dropping adaln_proj weights")
        t_c = time.time()
        import mlx.core as _mx_c
        active_before = _mx_c.get_active_memory() / 1024**3
        cache = pipe.build_adaln_cache_and_drop(
            num_steps=args.num_steps,
            has_visual_cond=bool(ref_image_paths or ref_video_paths),
            has_audio_cond=bool(ref_audio_paths),
            verbose=True,
        )
        active_after = _mx_c.get_active_memory() / 1024**3
        _log(f"adaln cache: {cache.num_steps} steps, "
             f"{cache.nbytes()/1e6:.1f} MB lookup table")
        _log(f"active_mlx {active_before:.2f} GiB -> {active_after:.2f} GiB "
             f"in {time.time()-t_c:.1f}s, {_mem_snapshot()}")

    # v16 260807 Sub3: install materialisation-barrier cadence on the DiT.
    pipe.dit._eval_every = int(args.eval_every)
    _log(f"materialisation barrier: mx.eval(h) every {args.eval_every} blocks "
         f"({'disabled' if args.eval_every == 0 else 'enabled'})")

    if args.layer_group_size > 0:
        _log(f"enabling LayerGroupManager (group_size={args.layer_group_size})")
        t_g = time.time()
        import mlx.core as mx
        active_before = mx.get_active_memory() / 1024**3
        mgr = pipe.enable_layer_group_eviction(group_size=args.layer_group_size,
                                                verbose=True)
        active_after = mx.get_active_memory() / 1024**3
        _log(f"LayerGroupManager installed in {time.time()-t_g:.1f}s, "
             f"active_mlx {active_before:.2f} GiB -> {active_after:.2f} GiB, "
             f"dormant={mgr.stats()['dormant_gb']:.2f} GiB, "
             f"{_mem_snapshot()}")

    # ---------- Stage 3: refs (image + audio) ----------
    import mlx.core as mx

    # v17 multiref: encode every image ref into ref_image_latents list.
    ref_image_latents = []
    if ref_image_paths:
        from PIL import Image
        for idx, rp in enumerate(ref_image_paths):
            img = Image.open(rp).convert("RGB").resize(
                (args.width, args.height), Image.LANCZOS)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = arr * 2.0 - 1.0
            arr = arr.transpose(2, 0, 1)[None, :, None, :, :]
            ref_x = mx.array(arr)
            zi = pipe.video_vae.encode(ref_x)
            ref_image_latents.append(zi)
            _log(f"ref_image_latents[{idx}] ({rp.name}) shape: {zi.shape}")
    ref_image_latent = ref_image_latents[0] if ref_image_latents else None

    ref_video_latents = []
    if ref_video_paths:
        import subprocess
        for idx, rvp in enumerate(ref_video_paths):
            # Decode video to raw rgb24 via ffmpeg.
            proc = subprocess.run(
                ["ffprobe","-v","error","-select_streams","v:0",
                 "-show_entries","stream=width,height,nb_frames,r_frame_rate",
                 "-of","default=nw=1:nk=1", str(rvp)],
                capture_output=True, text=True, check=True)
            pw, ph, nfr, rfr = proc.stdout.strip().split("\n")[:4]
            pw, ph = int(pw), int(ph)
            _log(f"ref_video[{idx}] probe ({rvp.name}): {pw}x{ph}, "
                 f"nb_frames={nfr}, fps={rfr}")
            rw = (pw // 16) * 16
            rh = (ph // 16) * 16
            raw = subprocess.run(
                ["ffmpeg","-loglevel","error","-i",str(rvp),
                 "-vf",f"scale={rw}:{rh},fps=24","-vframes","17",
                 "-pix_fmt","rgb24","-f","rawvideo","-"],
                capture_output=True, check=True).stdout
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, rh, rw, 3)
            _log(f"ref_video[{idx}] decoded frames={arr.shape[0]} at {rh}x{rw}")
            T = 17
            if arr.shape[0] < T:
                arr = np.concatenate(
                    [arr, np.repeat(arr[-1:], T - arr.shape[0], axis=0)], axis=0)
            else:
                arr = arr[:T]
            arr_f = arr.astype(np.float32) / 255.0
            arr_f = arr_f * 2.0 - 1.0
            arr_f = arr_f.transpose(3, 0, 1, 2)[None]
            ref_v_x = mx.array(arr_f)
            zv = pipe.video_vae.encode(ref_v_x)
            ref_video_latents.append(zv)
            _log(f"ref_video_latents[{idx}] shape: {zv.shape}")
    ref_video_latent = ref_video_latents[0] if ref_video_latents else None

    ref_audio_latents = []
    if ref_audio_paths:
        from scipy.io import wavfile
        for idx, rap in enumerate(ref_audio_paths):
            sr, wav = wavfile.read(rap)
            if wav.ndim == 1:
                wav = np.stack([wav, wav], axis=-1)
            wav = wav.astype(np.float32) / 32768.0
            wav_mx = mx.array(wav.T[None, ...])
            za = pipe.audio_vae.encode(wav_mx)
            ref_audio_latents.append(za)
            _log(f"ref_audio_latents[{idx}] ({rap.name}) shape: {za.shape}")
    ref_audio_latent = ref_audio_latents[0] if ref_audio_latents else None

    # v16 260807 Sub4: turn on phase-evict AFTER refs are encoded so we don't
    # trip the "video_vae was evicted" check inside pipe.video_vae.encode above.
    if args.phase_evict:
        _log("enabling phase-evict (video_vae will be dropped before denoise, "
             "reloaded before decode; DiT dropped before decode)")
        pipe.enable_phase_evict(evict_video_vae_for_denoise=True, verbose=True)

    # ---------- Stage 4: generate ----------
    _log(f"generating: {args.width}x{args.height}, {args.length} frames, "
         f"{args.num_steps} steps, seed={args.seed}")
    ctx_mx = mx.array(ctx_np).astype(mx.float32)
    t0 = time.time()
    video_np, audio_np, info = pipe.generate(
        prompt=args.prompt,  # ignored because context= is provided
        width=args.width, height=args.height, length=args.length,
        num_steps=args.num_steps, seed=args.seed,
        ref_image_latents=ref_image_latents,
        ref_video_latents=ref_video_latents,
        ref_audio_latents=ref_audio_latents,
        verbose=True,
        context=ctx_mx,
    )
    _log(f"generate done in {time.time()-t0:.1f}s, info={info}")
    try:
        _log(f"mlx peak={mx.get_peak_memory()/1024**3:.2f} GiB, "
             f"active={mx.get_active_memory()/1024**3:.2f} GiB")
    except Exception:
        pass

    # ---------- Stage 5: mux ----------
    out_path = Path(args.output).expanduser()
    _log(f"muxing to {out_path}...")
    _run_ffmpeg(video_np, audio_np, fps=24, sample_rate=32000, out_path=out_path)
    _log(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    _log(f"mem(final): {_mem_snapshot()}")


if __name__ == "__main__":
    main()
