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
    ap.add_argument("--ref-audio", default=None)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=576)
    ap.add_argument("--length", type=int, default=33)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--encoder-path",
                    default="~/mlx-video/mlx-models/H3-TextEncoder-MLX-bf16")
    ap.add_argument("--model-root",
                    default="~/mlx-video/mlx-models/MiniMaxH3-Ref2VA-MLX-Q4")
    ap.add_argument("--output", required=True)
    ap.add_argument("--context-npy", default=None,
                    help="Optional: skip encode step, load a pre-saved context")
    ap.add_argument("--save-context", default=None,
                    help="Also save the encoded context to this .npy path")
    ap.add_argument("--dry-encode", action="store_true",
                    help="Just encode and save context; skip DiT")
    args = ap.parse_args()

    _log(f"peak_rss(start)={_peak_rss_gb():.2f} GB")

    ref_image_path = Path(args.ref_image).expanduser() if args.ref_image else None
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
             f"has_ref_audio={ref_audio_path is not None}")
        t0 = time.time()
        ctx = enc.encode(args.prompt,
                         has_ref_image=ref_image_path is not None,
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

    ref_audio_latent = None
    if ref_audio_path:
        from scipy.signal import resample_poly
        # soundfile handles wav+mp3+ogg+flac; falls back to wavfile if missing
        try:
            import soundfile as sf
            wav, sr = sf.read(str(ref_audio_path), dtype='float32', always_2d=False)
        except (ImportError, RuntimeError):
            from scipy.io import wavfile
            sr, wav = wavfile.read(ref_audio_path)
        vae_sr = int(getattr(pipe.audio_vae, "sample_rate", 32000))
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        elif wav.dtype == np.int32:
            wav = wav.astype(np.float32) / 2147483648.0
        else:
            wav = wav.astype(np.float32)
        if sr != vae_sr:
            # Match ComfyUI comfy_extras/nodes_minimax_h3.py:_encode_ref_audio
            from math import gcd
            g = gcd(sr, vae_sr)
            wav = resample_poly(wav, vae_sr // g, sr // g, axis=0)
            _log(f"resampled ref audio {sr}Hz -> {vae_sr}Hz  (new samples={wav.shape[0]})")
        if wav.ndim == 1:
            wav = np.stack([wav, wav], axis=-1)
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
        ref_audio_latent=ref_audio_latent,
        verbose=True,
        context=ctx_mx,
    )
    _log(f"generate done in {time.time()-t0:.1f}s, info={info}")

    # ---------- Stage 5: mux ----------
    out_path = Path(args.output).expanduser()
    _log(f"muxing to {out_path}...")
    _run_ffmpeg(video_np, audio_np, fps=24, sample_rate=32000, out_path=out_path)
    _log(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    _log(f"peak_rss(final)={_peak_rss_gb():.2f} GB")


if __name__ == "__main__":
    main()
