"""Phase 8.11-3 diagnostic: FFT of a DiT-produced latent (dumped by
:code:`pipeline.generate(..., dump_latent_path=...)`).

Latent layout: NCDHW ``(B, 24, T_lat, H_lat, W_lat)`` on disk (np.save).

The task hypothesis:
    - If the ~16-px spatial grid in decoded RGB is upstream of the VAE, the
      DiT latent already shows periodic peaks at ``fx = 1/(16 / vae_ratio) = 1``
      cycles-per-latent-pixel (i.e. the Nyquist of the latent). Since our
      latent is at 1/16 pixel resolution, a 16-px pixel-space pattern maps to
      a 1-latent-pixel pattern (the Nyquist), which shows up as a peak at the
      edge of the FFT spectrum.

    - If the DiT latent is *clean* (no Nyquist peak beyond noise floor), the
      grid is 100% VAE-decoder-side. Prints the verdict either way.

Usage::

    python scripts/h3/analyze_dit_latent.py ~/tmp/h3_phase811_v3_latent.npy

Optionally pass ``--pixel-fft <mp4-path>`` to also FFT the decoded video
frames at the 16-px pixel band, for a full latent-vs-pixel comparison.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def latent_fft_report(latent_ncdhw: np.ndarray) -> None:
    """Print FFT peak / floor stats for a NCDHW latent tensor.

    We compute the 2D spatial FFT for each (b, c, t) plane, take the modulus,
    and look for peaks at the highest spatial frequency (period 2 latent-px =
    32 px in pixel space) and neighbouring bands. A clean latent has flat
    spectrum; a grid-y latent shows peaks at the edges.
    """
    B, C, T, H_lat, W_lat = latent_ncdhw.shape
    print(f"[fft] latent shape: B={B}, C={C}, T={T}, H_lat={H_lat}, W_lat={W_lat}")
    print(f"[fft] value stats: mean={latent_ncdhw.mean():.4f}, "
          f"std={latent_ncdhw.std():.4f}, "
          f"|max|={np.abs(latent_ncdhw).max():.4f}")

    # Flatten across (b, c, t) into "planes" of shape (H_lat, W_lat) and
    # aggregate FFT magnitudes.
    flat = latent_ncdhw.reshape(-1, H_lat, W_lat)
    Y = flat - flat.mean(axis=(1, 2), keepdims=True)
    F = np.fft.fftshift(np.fft.fft2(Y, axes=(1, 2)), axes=(1, 2))
    magF = np.abs(F).mean(axis=0)  # (H_lat, W_lat) averaged over planes

    Hc, Wc = H_lat // 2, W_lat // 2

    # peak at Nyquist edges (which maps back to 16-px pixel period if
    # H_lat = H/16). We want to know if there's a peak at kh=H_lat/2 or
    # kw=W_lat/2 (Nyquist).
    def peak_ratio_1d(spec_1d: np.ndarray, target_idx: int, window: int = 3) -> float:
        n = len(spec_1d)
        target_idx = int(target_idx)
        if target_idx <= 0 or target_idx >= n - 1:
            # can't compute at edge; return magnitude ratio to global mean
            return float(spec_1d[target_idx] / (spec_1d.mean() + 1e-9))
        lo = max(0, target_idx - window)
        hi = min(n, target_idx + window + 1)
        neighbors = np.concatenate([spec_1d[lo:target_idx], spec_1d[target_idx + 1:hi]])
        return float(spec_1d[target_idx] / (neighbors.mean() + 1e-9))

    col_spec = magF.sum(axis=0)
    row_spec = magF.sum(axis=1)

    # Sweep possible pixel-space periods (in the decoded output). For a
    # pixel-space period ``p_px``, the latent-space period is ``p_px / 16``,
    # and the corresponding freq index is ``W_lat / (p_px / 16) = W_lat * 16 / p_px``.
    print("\n[fft] pixel-space grid diagnostic (peak ratio at fx=1/p_px):")
    print(f"  {'period_px':>10} {'fx_col_idx':>10} {'col_ratio':>10} "
          f"{'fy_row_idx':>10} {'row_ratio':>10}")
    for p_px in (16, 32, 48, 64, 96, 128, 192):
        # latent-space cycles-per-image at pixel period p_px
        k_col_cycles = (W_lat * 16.0) / p_px
        k_row_cycles = (H_lat * 16.0) / p_px
        if k_col_cycles > W_lat // 2 or k_row_cycles > H_lat // 2:
            # would alias below Nyquist -- skip
            continue
        col_idx = int(round(Wc + k_col_cycles))
        row_idx = int(round(Hc + k_row_cycles))
        col_r = peak_ratio_1d(col_spec, col_idx)
        row_r = peak_ratio_1d(row_spec, row_idx)
        print(f"  {p_px:>10d} {col_idx:>10d} {col_r:>10.3f} "
              f"{row_idx:>10d} {row_r:>10.3f}")

    # Overall Nyquist test — if the highest-freq band has systematic peak,
    # the DiT is producing high-frequency energy that will alias.
    col_nyquist_mean = float(col_spec[[0, W_lat - 1]].mean())
    col_dc = float(col_spec[Wc])
    col_mid = float(col_spec[Wc - 3:Wc + 4].mean()) - col_dc / 7
    print(f"\n[fft] col spectrum: Nyquist edge mean={col_nyquist_mean:.4e}, "
          f"mid-band (excl DC)={col_mid:.4e}, "
          f"ratio nyq/mid={col_nyquist_mean/(col_mid+1e-9):.3f}")


def video_fft_report(mp4_path: Path) -> None:
    """Extract every frame, luma-FFT, report peak-at-1/16 ratio (col & row)."""
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp4_path), "-vf", "fps=8",
             os.path.join(td, "f_%04d.png")],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        from PIL import Image

        ratios_col: list[float] = []
        ratios_row: list[float] = []
        ratios_2d: list[float] = []
        for f in sorted(os.listdir(td)):
            if not f.endswith(".png"):
                continue
            arr = np.asarray(Image.open(os.path.join(td, f)).convert("RGB")).astype(
                np.float64
            ) / 255.0
            H, W, _ = arr.shape
            Y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            Y_dc = Y - Y.mean()
            F = np.fft.fftshift(np.fft.fft2(Y_dc))
            magF = np.abs(F)
            Wc = W // 2
            Hc = H // 2
            kw = W // 16
            kh = H // 16
            col_spec = magF.sum(axis=0)
            row_spec = magF.sum(axis=1)
            def _peak(spec_1d, target_idx, window=3):
                n = len(spec_1d)
                lo = max(0, target_idx - window)
                hi = min(n, target_idx + window + 1)
                nb = np.concatenate(
                    [spec_1d[lo:target_idx], spec_1d[target_idx + 1:hi]]
                )
                return spec_1d[target_idx] / (nb.mean() + 1e-9)
            ratios_col.append(_peak(col_spec, Wc + kw))
            ratios_row.append(_peak(row_spec, Hc + kh))
            p2d = magF[Hc + kh, Wc + kw]
            m2d = magF[Hc + kh - 2:Hc + kh + 3, Wc + kw - 2:Wc + kw + 3].mean()
            ratios_2d.append(p2d / (m2d + 1e-9))

        if ratios_col:
            print(f"[video-fft] {mp4_path.name}: "
                  f"col={np.mean(ratios_col):.3f} "
                  f"row={np.mean(ratios_row):.3f} "
                  f"2d(1/16, 1/16)={np.mean(ratios_2d):.3f} "
                  f"(n_frames={len(ratios_col)})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("latent_npy", help=".npy dump from pipeline.generate(dump_latent_path=)")
    p.add_argument("--pixel-fft", default=None, help="Also FFT the decoded mp4 at 1/16 band")
    args = p.parse_args()

    latent = np.load(args.latent_npy)
    latent_fft_report(latent)

    if args.pixel_fft:
        video_fft_report(Path(args.pixel_fft).expanduser())


if __name__ == "__main__":
    main()
