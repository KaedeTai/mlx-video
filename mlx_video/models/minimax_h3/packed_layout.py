"""PackedLayout: the [text | cond | audio | video] token-order builder.

Port of ``PackedLayout`` in ``comfy/ldm/minimax/model.py`` (lines 281-378).

Signatures:
- Task t2va: [text | audio | video]
- Task fl2va: [text | keyframe_cond_frame(s) | audio | video]
- Task ref2va: [text | (ref_img | ref_audio | ref_video_audio + ref_video)... | audio | video]

Position IDs are float64 [S, 3] over (t, h, w) in area-normalized axes:
`_axis_from_sqrt_area(dim, patch, sqrt_area)` scales into [0, 32] chunks;
temporal cursor uses `FRAME_RESCALE=5/3 * FRAME_PER_TOKEN[k%5]` spans.

All layout math is done in numpy (float64) since the sequence is small and
these tensors are consumed by the RoPE builder (numpy → mx conversion).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


FRAME_PER_TOKEN: Tuple[int, int, int, int, int] = (1, 4, 4, 4, 4)
FRAME_RESCALE: float = 5.0 / 3.0
VISUAL_COND_TIMESTEP: float = 0.999
AUDIO_COND_TIMESTEP: float = 1.0


def _axis_from_sqrt_area(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    """Area-normalized coordinate axis (float64), returning length ``dim // patch``.

    Matches ``_axis_from_sqrt_area`` in the reference:
      ``(arange(n) * (ratio / n) + (1 - ratio) / 2) * 32``.
    """
    ratio = dim / sqrt_area
    n = dim // patch
    return (np.arange(n, dtype=np.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Area-normalized (h, w) coordinate rows of one latent frame's 2×2 patches."""
    area = math.sqrt(h * w)
    hh, ww = np.meshgrid(_axis_from_sqrt_area(h, 2, area), _axis_from_sqrt_area(w, 2, area), indexing="ij")
    return np.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1), _axis_from_sqrt_area(w, 2, area)


def _video_t_spans(n: int) -> List[float]:
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(n)]


def _video_t_grid(n: int, origin: float) -> np.ndarray:
    spans = np.array(_video_t_spans(n), dtype=np.float64)
    return float(origin) + np.concatenate([np.zeros(1, dtype=np.float64), spans[:-1].cumsum(0)])


def _audio_grid(cursor: float, t: int, w_low: float, w_high: float) -> np.ndarray:
    """Channel-major stereo rows: t advances per latent frame, w pinned to the frame grid extremes."""
    g = np.zeros((t * 2, 3), dtype=np.float64)
    g[:, 0] = np.concatenate([cursor + np.arange(t, dtype=np.float64)] * 2)
    g[:t, 2] = w_low
    g[t:, 2] = w_high
    return g


def _video_grid(vt: int, frame: np.ndarray, cursor: float) -> np.ndarray:
    g = np.empty((vt, frame.shape[0], 3), dtype=np.float64)
    g[:, :, 0] = _video_t_grid(vt, cursor)[:, None]
    g[:, :, 1:] = frame[None]
    return g.reshape(-1, 3)


@dataclass
class RefBlock:
    """One reference block for ref2va layouts.

    ``kind`` = "image" | "audio" | "video" | "video_audio".
    """
    kind: str
    latent_h: int = 0
    latent_w: int = 0
    latent_t: int = 0
    ref_audio_t: int = 0


@dataclass
class Keyframe:
    """One resolved keyframe for fl2va (0 = first frame, ``frame_count-1`` = last)."""
    resolved_frame_index: int


class PackedLayout:
    """Static packed-sequence structure for one shape/conditioning signature."""

    def __init__(
        self,
        text_len: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        keyframes: Optional[Sequence[Keyframe]] = None,
        refs: Optional[Sequence[RefBlock]] = None,
        frame_count: Optional[int] = None,
    ):
        frame, w_grid = _frame_grid(latent_h, latent_w)
        frame_rows = frame.shape[0]

        segments: List[Tuple[str, int]] = [("text", text_len)]
        g = np.zeros((text_len, 3), dtype=np.float64)
        g[:, 0] = np.arange(text_len, dtype=np.float64)
        pos: List[np.ndarray] = [g]

        img_pos: List[np.ndarray] = []
        img_update: List[np.ndarray] = []
        audio_pos: List[np.ndarray] = []
        audio_update: List[np.ndarray] = []
        row = text_len

        # fl2va: keyframe cond rows right after text, sharing the target spatial grid
        if keyframes:
            for kf in keyframes:
                pixel_index = kf.resolved_frame_index
                if pixel_index == 0:
                    cond_t = float(text_len)
                elif frame_count is not None and pixel_index == frame_count - 1:
                    cond_t = float(text_len) + sum(_video_t_spans(latent_t)) - FRAME_RESCALE
                else:
                    raise ValueError("only first/last keyframe anchors are supported")
                g = np.empty((frame_rows, 3), dtype=np.float64)
                g[:, 0] = cond_t
                g[:, 1:] = frame
                segments.append(("cond", frame_rows))
                pos.append(g)
                img_pos.append(np.arange(row, row + frame_rows))
                img_update.append(np.zeros(frame_rows, dtype=bool))
                row += frame_rows

        target_audio_w = (float(w_grid[0]), float(w_grid[-1]))

        if refs:
            # Phase 9.0 (port ref_blocks layout from pipenetwork): image + audio
            # refs share a single rotary origin (== text_len). Neither image nor
            # audio advances the "block origin" independently; the target-audio /
            # target-video cursor advances by the *audio* span only (mirroring
            # pipenetwork FL2VA cond+audio-ref layout: cond is at fixed anchor,
            # only ref-audio latents contribute to the shared cursor advance).
            # This matches the FL2VA-native attention pattern; the pre-patch
            # behaviour shifted audio by +1 relative to image which broke ref-audio
            # content influence on generated speech (verified on pipenetwork:
            # F0=153Hz after this fix vs unvoiced/noise without).
            ref_origin = float(text_len)
            cursor = ref_origin
            for blk in refs:
                if blk.kind == "image":
                    r_frame, _ = _frame_grid(blk.latent_h, blk.latent_w)
                    n = r_frame.shape[0]
                    g = np.empty((n, 3), dtype=np.float64)
                    g[:, 0] = ref_origin  # shared origin (not sequential cursor)
                    g[:, 1:] = r_frame
                    segments.append(("ref_img", n))
                    pos.append(g)
                    img_pos.append(np.arange(row, row + n))
                    img_update.append(np.zeros(n, dtype=bool))
                    row += n
                    # image ref does NOT advance the shared cursor
                elif blk.kind == "audio":
                    rt = blk.ref_audio_t
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        # audio starts at shared origin, spans rt latent frames
                        pos.append(_audio_grid(ref_origin, rt, *target_audio_w))
                        audio_pos.append(np.arange(row, row + rt * 2))
                        audio_update.append(np.zeros(rt * 2, dtype=bool))
                        row += rt * 2
                    # only audio advances the shared cursor (matches pipenetwork)
                    cursor = ref_origin + float(rt)
                elif blk.kind in ("video", "video_audio"):
                    rt = blk.ref_audio_t
                    vt = blk.latent_t
                    r_frame, r_w_grid = _frame_grid(blk.latent_h, blk.latent_w)
                    if rt > 0:
                        segments.append(("ref_audio", rt * 2))
                        pos.append(_audio_grid(cursor, rt, float(r_w_grid[0]), float(r_w_grid[-1])))
                        audio_pos.append(np.arange(row, row + rt * 2))
                        audio_update.append(np.zeros(rt * 2, dtype=bool))
                        row += rt * 2
                    n = vt * r_frame.shape[0]
                    segments.append(("ref_img", n))
                    pos.append(_video_grid(vt, r_frame, cursor))
                    img_pos.append(np.arange(row, row + n))
                    img_update.append(np.zeros(n, dtype=bool))
                    row += n
                    cursor += max(float(rt), sum(_video_t_spans(vt)))
                else:
                    raise ValueError(f"unknown ref block kind: {blk.kind}")

        cursor = float(text_len) if not refs else cursor  # noqa: F823 (cursor defined above when refs, else initial value below)
        if not refs and not keyframes:
            cursor = float(text_len)

        # target audio then target video, always the last two segments
        segments.append(("audio", audio_t * 2))
        pos.append(_audio_grid(cursor, audio_t, *target_audio_w))
        audio_pos.append(np.arange(row, row + audio_t * 2))
        audio_update.append(np.ones(audio_t * 2, dtype=bool))
        row += audio_t * 2

        n_video = latent_t * frame_rows
        segments.append(("video", n_video))
        pos.append(_video_grid(latent_t, frame, cursor))
        img_pos.append(np.arange(row, row + n_video))
        img_update.append(np.ones(n_video, dtype=bool))
        row += n_video

        self.seq_len = row
        self.position_ids: np.ndarray = np.concatenate(pos, axis=0)  # [S, 3] float64
        self.img_pos: np.ndarray = np.concatenate(img_pos) if img_pos else np.zeros(0, dtype=np.int64)
        self.img_update: np.ndarray = np.concatenate(img_update) if img_update else np.zeros(0, dtype=bool)
        self.audio_pos: np.ndarray = np.concatenate(audio_pos) if audio_pos else np.zeros(0, dtype=np.int64)
        self.audio_update: np.ndarray = np.concatenate(audio_update) if audio_update else np.zeros(0, dtype=bool)
        self.signature = (text_len, latent_t, latent_h, latent_w, audio_t)

        # contiguous segment table (start, stop, kind)
        seg_abs: List[Tuple[int, int, str]] = []
        off = 0
        for kind, n in segments:
            seg_abs.append((off, off + n, kind))
            off += n
        self.segments: List[Tuple[int, int, str]] = seg_abs

    def __repr__(self) -> str:  # pragma: no cover
        return (f"PackedLayout(seq_len={self.seq_len}, signature={self.signature}, "
                f"n_segments={len(self.segments)})")
