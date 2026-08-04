"""CLI entry point: `python -m mlx_video.models.minimax_h3.generate`.

Loads the full MLX H3 pipeline and produces a joint video+audio mp4 for a
user-supplied prompt (+ optional reference image / audio).

Uses ffmpeg via subprocess to mux the RGB frames + stereo waveform into an mp4.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def _run_ffmpeg(video_rgb: np.ndarray, audio_stereo: np.ndarray,
                fps: int, sample_rate: int, out_path: Path):
    """Mux [T, H, W, 3] uint8 frames + [2, N] fp32 waveform into an mp4."""
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, C = video_rgb.shape
    assert C == 3
    assert audio_stereo.shape[0] == 2

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_video = td / "video.raw"
        raw_audio = td / "audio.wav"

        # Video: raw RGB24
        raw_video.write_bytes(video_rgb.tobytes())

        # Audio: use scipy.io.wavfile if available, else raw
        # Interleave: [2, N] -> [N, 2]
        audio_i = audio_stereo.T.astype(np.float32)
        # Clip to [-1, 1] and convert to int16
        audio_int16 = (np.clip(audio_i, -1.0, 1.0) * 32767).astype(np.int16)
        try:
            from scipy.io import wavfile
            wavfile.write(str(raw_audio), sample_rate, audio_int16)
        except ImportError:
            import wave
            with wave.open(str(raw_audio), "wb") as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(fps), "-i", str(raw_video),
            "-i", str(raw_audio),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="A person speaking to camera warmly.")
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--length", type=int, default=5, help="frame count (snaps to 17k+5 grid)")
    p.add_argument("--num-steps", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model-root", default="~/mlx-video/mlx-models/MiniMaxH3-Ref2VA-MLX-bf16")
    p.add_argument("--output", default="~/tmp/h3_mlx_smoke_test.mp4")
    p.add_argument("--ref-image", default=None, help="path to reference image")
    p.add_argument("--ref-audio", default=None, help="path to reference audio (wav)")
    args = p.parse_args()

    from .pipeline import load_pipeline
    from .video_vae import IMAGENET_MEAN, IMAGENET_STD

    print(f"[generate] loading pipeline from {args.model_root}...")
    pipe = load_pipeline(Path(args.model_root))
    print("[generate] pipeline loaded")

    ref_image_latent = None
    if args.ref_image:
        import mlx.core as mx
        try:
            from PIL import Image
        except ImportError:
            raise SystemExit("PIL required for --ref-image")
        img = Image.open(args.ref_image).convert("RGB").resize((args.width, args.height))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0,1]
        # normalize per imagenet, then reshape to NCHW [-1,1] via pipeline convention
        arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        # video_vae expects [B, C, T, H, W]
        arr = arr.transpose(2, 0, 1)[None, :, None, :, :]  # (1,3,1,H,W)
        ref_x = mx.array(arr)
        ref_image_latent = pipe.video_vae.encode(ref_x)
        print(f"[generate] ref_image_latent shape: {ref_image_latent.shape}")

    ref_audio_latent = None
    if args.ref_audio:
        import mlx.core as mx
        try:
            from scipy.io import wavfile
        except ImportError:
            raise SystemExit("scipy required for --ref-audio")
        sr, wav = wavfile.read(args.ref_audio)
        if wav.ndim == 1:
            wav = np.stack([wav, wav], axis=-1)
        wav = wav.astype(np.float32) / 32768.0
        # audio_vae expects [1, C=2, T]
        wav_mx = mx.array(wav.T[None, ...])
        ref_audio_latent = pipe.audio_vae.encode(wav_mx)
        print(f"[generate] ref_audio_latent shape: {ref_audio_latent.shape}")

    print(f"[generate] sampling {args.length} frames at {args.width}x{args.height}, "
          f"{args.num_steps} steps, seed={args.seed}...")

    video_np, audio_np, info = pipe.generate(
        prompt=args.prompt,
        width=args.width, height=args.height, length=args.length,
        num_steps=args.num_steps, seed=args.seed,
        ref_image_latent=ref_image_latent,
        ref_audio_latent=ref_audio_latent,
        verbose=True,
    )

    print(f"[generate] done: {info}")

    out_path = Path(args.output).expanduser()
    print(f"[generate] muxing to {out_path}...")
    _run_ffmpeg(video_np, audio_np, fps=24, sample_rate=32000, out_path=out_path)
    print(f"[generate] wrote {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
