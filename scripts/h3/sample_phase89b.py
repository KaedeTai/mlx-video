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


def _peak_rss_gb() -> float:
    return _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024**3


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
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(fps), "-i", str(raw_video),
            "-i", str(raw_audio),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k", "-shortest",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--ref-image", default=None)
    ap.add_argument("--ref-video", default=None,
                    help="Path to a reference video (mp4). Encoded via video_vae and used as kind='video' RefBlock.")
    ap.add_argument("--ref-audio", default=None)
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
    ap.add_argument("--layer-group-size", type=int, default=0,
                    help="If >0, enable LayerGroupManager on the DiT blocks. "
                         "Weights of dormant groups live in CPU RAM (not Metal-wired), "
                         "trading ~+30-40 ms/step for a lower peak Metal working set.")
    args = ap.parse_args()

    _log(f"peak_rss(start)={_peak_rss_gb():.2f} GB")

    ref_image_path = Path(args.ref_image).expanduser() if args.ref_image else None
    ref_video_path = Path(args.ref_video).expanduser() if args.ref_video else None
    ref_audio_path = Path(args.ref_audio).expanduser() if args.ref_audio else None

    # ---------- Stage 1: text encoder ----------
    if args.context_npy is None:
        from mlx_video.models.minimax_h3.text_encoder_bridge import H3TextEncoderBridge
        import mlx.core as mx

        _log("stage 1: loading H3TextEncoderBridge...")
        t0 = time.time()
        enc = H3TextEncoderBridge(model_path=args.encoder_path)
        _log(f"encoder loaded in {time.time()-t0:.1f}s, "
             f"peak_rss={_peak_rss_gb():.2f} GB")

        _log(f"encoding prompt: {args.prompt!r}")
        _log(f"  has_ref_image={ref_image_path is not None}, "
             f"has_ref_video={ref_video_path is not None}, "
             f"has_ref_audio={ref_audio_path is not None}")
        t0 = time.time()
        ctx = enc.encode(args.prompt,
                         has_ref_image=ref_image_path is not None,
                         has_ref_video=ref_video_path is not None,
                         has_ref_audio=ref_audio_path is not None)
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
        _log(f"encoder freed, peak_rss={_peak_rss_gb():.2f} GB")
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
    _log(f"pipeline loaded in {time.time()-t0:.1f}s, "
         f"peak_rss={_peak_rss_gb():.2f} GB")

    if args.turbo_lora:
        from mlx_video.models.minimax_h3.lora import load_turbo_lora
        _log(f"loading Turbo LoRA: {args.turbo_lora} (alpha={args.turbo_lora_alpha})")
        t_l = time.time()
        n_wrapped = load_turbo_lora(pipe.dit, args.turbo_lora,
                                     alpha=args.turbo_lora_alpha, verbose=True)
        _log(f"Turbo LoRA installed: {n_wrapped} modules in {time.time()-t_l:.1f}s, "
             f"peak_rss={_peak_rss_gb():.2f} GB")

    if args.layer_group_size > 0:
        _log(f"enabling LayerGroupManager (group_size={args.layer_group_size})")
        t_g = time.time()
        import mlx.core as mx
        active_before = mx.get_active_memory() / 1024**3
        mgr = pipe.enable_layer_group_eviction(group_size=args.layer_group_size,
                                                verbose=True)
        active_after = mx.get_active_memory() / 1024**3
        _log(f"LayerGroupManager installed in {time.time()-t_g:.1f}s, "
             f"active_mlx {active_before:.2f} GB -> {active_after:.2f} GB, "
             f"dormant={mgr.stats()['dormant_gb']:.2f} GB, "
             f"peak_rss={_peak_rss_gb():.2f} GB")

    # ---------- Stage 3: refs (image + audio) ----------
    import mlx.core as mx

    ref_image_latent = None
    if ref_image_path:
        from PIL import Image
        img = Image.open(ref_image_path).convert("RGB").resize(
            (args.width, args.height), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        arr = arr.transpose(2, 0, 1)[None, :, None, :, :]
        ref_x = mx.array(arr)
        ref_image_latent = pipe.video_vae.encode(ref_x)
        _log(f"ref_image_latent shape: {ref_image_latent.shape}")

    ref_video_latent = None
    if ref_video_path:
        import subprocess
        # Decode video to raw rgb24 via ffmpeg
        proc = subprocess.run(
            ["ffprobe","-v","error","-select_streams","v:0",
             "-show_entries","stream=width,height,nb_frames,r_frame_rate",
             "-of","default=nw=1:nk=1", str(ref_video_path)],
            capture_output=True, text=True, check=True)
        pw, ph, nfr, rfr = proc.stdout.strip().split("\n")[:4]
        pw, ph = int(pw), int(ph)
        _log(f"ref_video probe: {pw}x{ph}, nb_frames={nfr}, fps={rfr}")
        # Use ref video native res (multiple of 16 required by vae ratio),
        # snap down to nearest 16 if needed.
        rw = (pw // 16) * 16
        rh = (ph // 16) * 16
        raw = subprocess.run(
            ["ffmpeg","-loglevel","error","-i",str(ref_video_path),
             "-vf",f"scale={rw}:{rh},fps=24","-vframes","17",
             "-pix_fmt","rgb24","-f","rawvideo","-"],
            capture_output=True, check=True).stdout
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, rh, rw, 3)
        _log(f"ref_video decoded frames={arr.shape[0]} at {rh}x{rw}")
        # pad/truncate to multiple of clip_length (17 for h3)
        T = 17
        if arr.shape[0] < T:
            arr = np.concatenate([arr, np.repeat(arr[-1:], T - arr.shape[0], axis=0)], axis=0)
        else:
            arr = arr[:T]
        arr_f = arr.astype(np.float32) / 255.0
        arr_f = arr_f * 2.0 - 1.0
        # [T, H, W, 3] -> [1, 3, T, H, W]
        arr_f = arr_f.transpose(3, 0, 1, 2)[None]
        ref_v_x = mx.array(arr_f)
        ref_video_latent = pipe.video_vae.encode(ref_v_x)
        _log(f"ref_video_latent shape: {ref_video_latent.shape}")

    ref_audio_latent = None
    if ref_audio_path:
        from scipy.io import wavfile
        sr, wav = wavfile.read(ref_audio_path)
        if wav.ndim == 1:
            wav = np.stack([wav, wav], axis=-1)
        wav = wav.astype(np.float32) / 32768.0
        wav_mx = mx.array(wav.T[None, ...])
        ref_audio_latent = pipe.audio_vae.encode(wav_mx)
        _log(f"ref_audio_latent shape: {ref_audio_latent.shape}")

    # ---------- Stage 4: generate ----------
    _log(f"generating: {args.width}x{args.height}, {args.length} frames, "
         f"{args.num_steps} steps, seed={args.seed}")
    ctx_mx = mx.array(ctx_np).astype(mx.float32)
    t0 = time.time()
    video_np, audio_np, info = pipe.generate(
        prompt=args.prompt,  # ignored because context= is provided
        width=args.width, height=args.height, length=args.length,
        num_steps=args.num_steps, seed=args.seed,
        ref_image_latent=ref_image_latent,
        ref_video_latent=ref_video_latent,
        ref_audio_latent=ref_audio_latent,
        verbose=True,
        context=ctx_mx,
    )
    _log(f"generate done in {time.time()-t0:.1f}s, info={info}")
    try:
        _log(f"mlx peak={mx.get_peak_memory()/1024**3:.2f} GB, "
             f"active={mx.get_active_memory()/1024**3:.2f} GB")
    except Exception:
        pass

    # ---------- Stage 5: mux ----------
    out_path = Path(args.output).expanduser()
    _log(f"muxing to {out_path}...")
    _run_ffmpeg(video_np, audio_np, fps=24, sample_rate=32000, out_path=out_path)
    _log(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    _log(f"peak_rss(final)={_peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
