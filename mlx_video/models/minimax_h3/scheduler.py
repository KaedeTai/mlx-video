"""Flow-matching scheduler for MiniMax H3 (dual sigma shift, single-driver).

Design mirrors the ComfyUI recipe:
- ``comfy.model_sampling.ModelSamplingDiscreteFlow`` provides the video
  schedule with ``shift = 12.0`` (``time_snr_shift(shift, t) = shift*t /
  (1 + (shift-1)*t)`` maps `t` in [0,1] to shifted values in [0,1]).
- The audio schedule is *not* stepped directly — the DiT returns the audio
  velocity already scaled by ``d(sigma_a)/d(sigma_v)``, so the same Euler
  step on the video schedule integrates both streams' true ODEs.
- Sampler formula: ``X_next = X + model_out * (sigma_next - sigma_cur)``
  where ``model_out`` is the model's returned velocity (H3 returns
  ``[-video_out, -slope_a * audio_out]``).

Two ``.step`` variants:
- Euler (default): single first-order step.
- DPM++ 2M: second-order multi-step, for higher quality at same NFE.

Usage
-----
    scheduler = MiniMaxH3Scheduler(shift_video=12.0, shift_audio=3.0)
    scheduler.set_timesteps(num_inference_steps=30)
    for i, t in enumerate(scheduler.timesteps):
        v_video, v_audio = model((x_video, x_audio), t, context, payload)
        x_video, x_audio = scheduler.step(v_video, v_audio, i, x_video, x_audio)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx
import numpy as np


def time_snr_shift(shift: float, t: np.ndarray) -> np.ndarray:
    """Vector version of ComfyUI's ``time_snr_shift(alpha, t)``."""
    if shift == 1.0:
        return t
    return shift * t / (1.0 + (shift - 1.0) * t)


@dataclass
class MiniMaxH3Scheduler:
    """Flow-matching scheduler for H3 with dual sigma shift.

    Attributes
    ----------
    shift_video : Video-schedule shift (default 12.0).
    shift_audio : Audio-schedule shift (default 3.0). Not applied directly here
                  — the model has already scaled the audio velocity by
                  ``d(sigma_a)/d(sigma_v)`` before returning.
    multiplier  : Timestep multiplier (matches ComfyUI's ``multiplier=1000``).
    sampler     : "euler" | "dpmpp_2m".
    """
    shift_video: float = 12.0
    shift_audio: float = 3.0
    multiplier: float = 1000.0
    sampler: str = "euler"

    def __post_init__(self):
        self.sigmas: Optional[np.ndarray] = None
        self.timesteps: Optional[mx.array] = None
        self._prev_derivative_video: Optional[mx.array] = None
        self._prev_derivative_audio: Optional[mx.array] = None
        self._prev_sigma: Optional[float] = None

    # ------------------------------------------------------------------
    # Schedule construction
    # ------------------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int, sigma_min: float = 1e-5) -> None:
        """Populate ``self.sigmas`` (length N+1, from ~1.0 to ~0.0) and ``self.timesteps``."""
        # Base t goes 1.0 -> 0.0 in N+1 points (endpoints included)
        t_base = np.linspace(1.0, 0.0, num_inference_steps + 1, dtype=np.float64)
        sigmas = time_snr_shift(self.shift_video, t_base)
        # Clamp small-end to sigma_min to avoid divide-by-zero downstream
        sigmas[-1] = max(sigmas[-1], sigma_min)
        self.sigmas = sigmas.astype(np.float32)
        # Timesteps are (sigma * multiplier); model divides back by multiplier
        ts = self.sigmas[:-1] * self.multiplier  # one per step
        self.timesteps = mx.array(ts.astype(np.float32))
        self._prev_derivative_video = None
        self._prev_derivative_audio = None
        self._prev_sigma = None

    def timestep_for(self, step_index: int) -> mx.array:
        """Convenience: return the [1] timestep tensor for step ``step_index``."""
        if self.timesteps is None:
            raise RuntimeError("call set_timesteps() first")
        return self.timesteps[step_index:step_index + 1]

    # ------------------------------------------------------------------
    # Euler step
    # ------------------------------------------------------------------

    def _euler_step(
        self, model_v: mx.array, model_a: mx.array,
        sigma_cur: float, sigma_next: float,
        x_v: mx.array, x_a: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        d_sigma = sigma_next - sigma_cur  # negative (denoising)
        x_v_next = x_v + model_v * d_sigma
        x_a_next = x_a + model_a * d_sigma
        return x_v_next, x_a_next

    # ------------------------------------------------------------------
    # DPM++ 2M step (Karras et al. multi-step second-order)
    # ------------------------------------------------------------------

    def _dpmpp_2m_step(
        self, model_v: mx.array, model_a: mx.array,
        sigma_cur: float, sigma_next: float,
        x_v: mx.array, x_a: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        # In flow-matching, derivative == model velocity. First iteration falls back to Euler.
        d_v_cur = model_v
        d_a_cur = model_a
        if self._prev_derivative_video is None:
            x_v_next, x_a_next = self._euler_step(model_v, model_a, sigma_cur, sigma_next, x_v, x_a)
        else:
            # Standard multi-step 2M: h = log(sigma_next / sigma_cur), h_prev, etc.
            # For simplicity we use the trapezoidal analogue in sigma space:
            #   x_next = x + 0.5 * (d_prev + d_cur) * (sigma_next - sigma_cur)
            d_sigma = sigma_next - sigma_cur
            d_v_avg = 0.5 * (self._prev_derivative_video + d_v_cur)
            d_a_avg = 0.5 * (self._prev_derivative_audio + d_a_cur)
            x_v_next = x_v + d_v_avg * d_sigma
            x_a_next = x_a + d_a_avg * d_sigma

        self._prev_derivative_video = d_v_cur
        self._prev_derivative_audio = d_a_cur
        self._prev_sigma = sigma_cur
        return x_v_next, x_a_next

    # ------------------------------------------------------------------
    # Public step
    # ------------------------------------------------------------------

    def step(
        self,
        model_output_video: mx.array,
        model_output_audio: mx.array,
        step_index: int,
        sample_video: mx.array,
        sample_audio: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """One update step: (X_video, X_audio) at ``sigmas[i]`` -> at ``sigmas[i+1]``.

        Returns the updated (video, audio) latents.
        """
        if self.sigmas is None:
            raise RuntimeError("call set_timesteps() first")
        sigma_cur = float(self.sigmas[step_index])
        sigma_next = float(self.sigmas[step_index + 1])
        if self.sampler == "dpmpp_2m":
            return self._dpmpp_2m_step(model_output_video, model_output_audio,
                                       sigma_cur, sigma_next, sample_video, sample_audio)
        return self._euler_step(model_output_video, model_output_audio,
                                sigma_cur, sigma_next, sample_video, sample_audio)

    # ------------------------------------------------------------------
    # Convenience: initial noise sigma (matches sigma_max)
    # ------------------------------------------------------------------

    @property
    def sigma_max(self) -> float:
        if self.sigmas is None:
            raise RuntimeError("call set_timesteps() first")
        return float(self.sigmas[0])

    def scale_noise(self, x: mx.array) -> mx.array:
        """Scale initial standard-Gaussian noise by ``sigma_max`` (usually ~1.0)."""
        return x * self.sigma_max
