"""3-axis (t, h, w) split-half rotary embedding for the packed H3 sequence.

Port of :func:`rope_rotation_table` and :meth:`MiniMaxH3Model.rope_freqs`
(``comfy/ldm/minimax/model.py`` lines 130-140 and 508-518).

Position-ID tensor is float64 [S, 3] over (t, h, w) in area-normalized axes.
Per-axis multiply by ``inv_freq [16]`` then concat (t | h | w) → [S, 48] pair
angles, then duplicated halves → [S, 96]; rotation table shape
[1, S, 1, 48, 2, 2].
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def compute_rope_angles(position_ids: np.ndarray, inv_freq: np.ndarray) -> np.ndarray:
    """Given a float64 [S, 3] position tensor and [F] inv_freq, produce the [S, 2*3*F] pair angles.

    Mirrors ``MiniMaxH3Model.rope_freqs``: per-axis multiply, concat (t|h|w) →
    [S, 3*F], then duplicate the whole thing (split-half convention) → [S, 6*F].
    """
    if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
        raise ValueError(f"position_ids must be [S, 3]; got {position_ids.shape}")
    pos_f32 = position_ids.astype(np.float32)
    # [S, 3, F] = [S, 3, 1] * [1, 1, F]  (all float32 to mirror the reference)
    per_axis = pos_f32[:, :, None] * inv_freq.astype(np.float32)[None, None, :]
    t_f, h_f, w_f = per_axis[:, 0, :], per_axis[:, 1, :], per_axis[:, 2, :]
    half = np.concatenate([t_f, h_f, w_f], axis=-1)  # [S, 3F]
    full = np.concatenate([half, half], axis=-1)     # [S, 6F]  duplicated
    return full


def rope_rotation_table(angles: np.ndarray, dtype: mx.Dtype = mx.float32) -> mx.array:
    """[S, rot_dim] pair angles → [1, S, 1, rot_dim/2, 2, 2] rotation matrices.

    The angles argument has duplicated halves (``angles[:, :half] == angles[:, half:]``).
    Only the first half is used to compute the rotation matrix.
    """
    S, rot_dim = angles.shape
    half = rot_dim // 2
    ang = angles[:, :half]
    c = np.cos(ang).astype(np.float32)
    s = np.sin(ang).astype(np.float32)
    table_np = np.stack([c, -s, s, c], axis=-1).reshape(1, S, 1, half, 2, 2)
    return mx.array(table_np).astype(dtype)


def build_rope_table(
    position_ids: np.ndarray,
    inv_freq: np.ndarray,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Convenience: position_ids [S, 3] + inv_freq [F] → rotation table [1, S, 1, 3*F, 2, 2]."""
    ang = compute_rope_angles(position_ids, inv_freq)
    return rope_rotation_table(ang, dtype=dtype)


def apply_split_half_rope(x: mx.array, table: mx.array, rot_dim: int) -> mx.array:
    """Apply split-half rotary to the first ``rot_dim`` dims of x's last axis.

    Args
    ----
    x     : [..., S, H, head_dim]     (leading axes may include a batch dim of size 1)
    table : [1, S, 1, rot_dim/2, 2, 2]
    rot_dim: even int, ≤ head_dim.

    The remaining head_dim - rot_dim tail dims pass through un-rotated.
    """
    half = rot_dim // 2
    head_dim = x.shape[-1]
    if head_dim < rot_dim:
        raise ValueError(f"head_dim {head_dim} < rot_dim {rot_dim}")

    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:] if head_dim > rot_dim else None

    # split-half: pair up (x[..., i], x[..., i+half]) for i in [0, half)
    x_a = x_rot[..., :half]
    x_b = x_rot[..., half:rot_dim]

    # table shape [1, S, 1, half, 2, 2]; extract cos/sin per position
    tbl = table[0, :, 0, :, :, :]  # [S, half, 2, 2]
    cos = tbl[..., 0, 0]           # [S, half]
    sin = tbl[..., 1, 0]           # [S, half]

    # Broadcast against [..., S, H, half]: insert head axis
    while cos.ndim < x_a.ndim:
        cos = mx.expand_dims(cos, axis=-2)
        sin = mx.expand_dims(sin, axis=-2)

    new_a = (x_a * cos.astype(x_a.dtype)) - (x_b * sin.astype(x_a.dtype))
    new_b = (x_a * sin.astype(x_a.dtype)) + (x_b * cos.astype(x_a.dtype))

    if x_pass is not None:
        return mx.concatenate([new_a, new_b, x_pass], axis=-1)
    return mx.concatenate([new_a, new_b], axis=-1)
