"""Phase 5 scheduler tests."""

import math

import mlx.core as mx
import numpy as np

from mlx_video.models.minimax_h3.scheduler import (
    MiniMaxH3Scheduler,
    time_snr_shift,
)
from mlx_video.models.minimax_h3.model import time_shift_sigma, time_shift_slope


def test_time_snr_shift_identity():
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    out = time_snr_shift(1.0, t)
    assert np.allclose(out, t)


def test_time_snr_shift_boundary():
    """At t=0 and t=1 the shift is a no-op."""
    for shift in (1.0, 3.0, 12.0, 100.0):
        assert time_snr_shift(shift, np.array([0.0]))[0] == 0.0
        assert abs(time_snr_shift(shift, np.array([1.0]))[0] - 1.0) < 1e-6


def test_set_timesteps():
    sched = MiniMaxH3Scheduler(shift_video=12.0)
    sched.set_timesteps(30)
    assert sched.sigmas.shape == (31,)
    assert sched.sigmas[0] >= 0.99  # ~1.0 with sigma_max clamp
    assert sched.sigmas[-1] <= 1e-4
    # Monotonically decreasing
    assert (np.diff(sched.sigmas) <= 0).all()
    assert sched.timesteps.shape == (30,)


def test_step_euler_denoises():
    """A trivial model that returns velocity = -x pushes x -> 0 across steps."""
    sched = MiniMaxH3Scheduler(shift_video=12.0)
    sched.set_timesteps(50)
    x_v = mx.array(np.ones((1, 4), dtype=np.float32))
    x_a = mx.array(np.ones((1, 4), dtype=np.float32))
    for i in range(50):
        # model returns -x (moves x toward 0 as d_sigma < 0: x_next = x + (-x)*d_sigma = x + |d_sigma|*x -- wait)
        # For the ideal noise-prediction case in flow matching,
        # velocity = x (positive) makes x_next = x + x*d_sigma with d_sigma<0 → x smaller.
        v_v = x_v * 1.0
        v_a = x_a * 1.0
        x_v, x_a = sched.step(v_v, v_a, i, x_v, x_a)
    # After 50 steps of decay, x should be much smaller
    assert float(mx.abs(x_v).sum()) < 2.0  # ~4 * exp(-1) ≈ 1.47
    assert float(mx.abs(x_a).sum()) < 2.0


def test_dual_shift_slope_property():
    """time_shift_slope should equal the numerical derivative d(sigma_a)/d(sigma_v)."""
    shift_v, shift_a = 12.0, 3.0
    for sigma_v in [0.1, 0.3, 0.5, 0.9]:
        eps = 1e-6
        sigma_a = time_shift_sigma(sigma_v, shift_v, shift_a)
        sigma_a_p = time_shift_sigma(sigma_v + eps, shift_v, shift_a)
        num_deriv = (sigma_a_p - sigma_a) / eps
        analytic = time_shift_slope(sigma_v, shift_v, shift_a)
        assert abs(num_deriv - analytic) < 1e-3, (sigma_v, num_deriv, analytic)


if __name__ == "__main__":
    test_time_snr_shift_identity()
    print("time_snr_shift identity OK")
    test_time_snr_shift_boundary()
    print("time_snr_shift boundary OK")
    test_set_timesteps()
    print("set_timesteps OK")
    test_step_euler_denoises()
    print("euler step denoises OK")
    test_dual_shift_slope_property()
    print("dual shift slope OK")
