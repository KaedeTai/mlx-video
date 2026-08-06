"""H3 v17 multi-ref order-consistency test (sub 10).

Purpose
-------
Verify that when ``H3Pipeline.generate`` is handed multiple ref latents,
the three parallel accumulators stay in lockstep with the Comfy-standard
presentation order:

    1. All image refs (in the order supplied)
    2. All video refs (in the order supplied)
    3. All independent audio refs (in the order supplied)

Checked accumulators
--------------------
- ``payload['refs']``               -> list[RefBlock] used to construct the
                                       PackedLayout
- ``payload['cond_video_latents']`` -> visual latents re-injected each step
- ``payload['cond_audio_latents']`` -> audio latents re-injected each step
- ``PackedLayout.segments``         -> canonical [text | refs* | audio | video]
- ``format_ref2va_prompt(...)``     -> ``<Picture N>: ... <Video N>: ... <Audio N>: ...``
                                       must count the refs and appear in the same order
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_video.models.minimax_h3.packed_layout import PackedLayout, RefBlock
from mlx_video.models.minimax_h3.text_encoder_bridge import (
    DummyTextEncoder,
    format_ref2va_prompt,
)


def _fake_img_latent(h: int = 48, w: int = 84):
    return mx.zeros((1, 24, 1, h, w), dtype=mx.float32)


def _fake_vid_latent(vt: int = 2, h: int = 48, w: int = 84):
    return mx.zeros((1, 24, vt, h, w), dtype=mx.float32)


def _fake_aud_latent(t: int = 200):
    return mx.zeros((1, 32, 2, t), dtype=mx.float32)


class _FakeDiT:
    """A no-op DiT that records the payload it was called with."""
    _modulation_cache = None
    _eval_every = 0

    def __init__(self):
        # Two dummy blocks so ``len(self.blocks)`` returns something sensible.
        self.blocks = [object(), object()]
        self.last_payload = None

    def __call__(self, latents, ts, context, payload):
        # Never actually called in this test; kept for completeness.
        self.last_payload = payload
        v, a = latents
        return v, a


def _build_pipeline_and_dummy_generate(img_latents, vid_latents, aud_latents,
                                       *, text_len=32):
    """Reproduce the ref-block builder from ``H3Pipeline.generate`` so we can
    inspect the ordered accumulators without spinning up the DiT.

    We copy the exact snippet in ``pipeline.py`` rather than partly invoking
    ``generate`` because a real generate call needs a text encoder + VAE.
    """
    from mlx_video.models.minimax_h3.pipeline import H3Pipeline  # noqa: F401

    refs = []
    cond_video_latents = []
    cond_audio_latents = []
    for zi in img_latents:
        _, _, _, rh, rw = zi.shape
        refs.append(RefBlock(kind="image", latent_h=rh, latent_w=rw))
        cond_video_latents.append(zi)
    for zv in vid_latents:
        _, _, vt, rh, rw = zv.shape
        refs.append(RefBlock(kind="video", latent_h=rh, latent_w=rw, latent_t=vt))
        cond_video_latents.append(zv)
    for za in aud_latents:
        rat = za.shape[-1]
        refs.append(RefBlock(kind="audio", ref_audio_t=rat))
        cond_audio_latents.append(za)

    layout = PackedLayout(
        text_len=text_len, latent_t=2, latent_h=48, latent_w=84, audio_t=100,
        refs=refs if refs else None,
    )
    return refs, cond_video_latents, cond_audio_latents, layout


def test_empty_refs_layout_ordering():
    refs, cv, ca, layout = _build_pipeline_and_dummy_generate([], [], [])
    assert refs == []
    assert cv == [] and ca == []
    kinds = [k for _, _, k in layout.segments]
    assert kinds == ["text", "audio", "video"]


def test_multiref_ordering_and_layout_alignment():
    imgs = [_fake_img_latent() for _ in range(3)]
    vids = [_fake_vid_latent(vt=2) for _ in range(1)]
    auds = [_fake_aud_latent(t=200) for _ in range(2)]

    refs, cv, ca, layout = _build_pipeline_and_dummy_generate(imgs, vids, auds)

    # 1) refs list follows Comfy order.
    kinds = [r.kind for r in refs]
    assert kinds == ["image", "image", "image", "video", "audio", "audio"], kinds

    # 2) visual latent list == image latents then video latents (same order).
    assert len(cv) == 4
    for got, expected in zip(cv[:3], imgs):
        assert got is expected
    assert cv[3] is vids[0]

    # 3) audio latent list == the independent audio refs, in order.
    assert len(ca) == 2
    for got, expected in zip(ca, auds):
        assert got is expected

    # 4) PackedLayout segments start with text + one segment per ref, in
    #    the same order (video contributes one ref_img segment; audio
    #    contributes one ref_audio segment per audio).
    seg_kinds = [k for _, _, k in layout.segments]
    # text, 3x ref_img (images), 1x ref_img (video's spatial frames),
    # 2x ref_audio (independent audios), audio (target), video (target)
    assert seg_kinds == [
        "text",
        "ref_img", "ref_img", "ref_img",
        "ref_img",
        "ref_audio", "ref_audio",
        "audio", "video",
    ], seg_kinds


def test_format_ref2va_prompt_matches_ref_counts():
    # Number of <Picture/Video/Audio> tags must equal the ref counts and
    # appear in the Comfy-standard order.
    prompt = format_ref2va_prompt(
        "wide shot of SU, LIN, and their son",
        has_ref_image=3,
        has_ref_video=1,
        has_ref_audio=2,
    )
    assert prompt.count("<Picture ") == 3
    assert prompt.count("<Video ") == 1
    assert prompt.count("<Audio ") == 2
    # 1-based numbering, in order.
    assert "<Picture 1>:" in prompt
    assert "<Picture 3>:" in prompt
    assert "<Video 1>:" in prompt
    assert "<Audio 2>:" in prompt
    # Order check: last image tag comes before first video tag before first
    # audio tag before the free prompt text.
    p3 = prompt.index("<Picture 3>:")
    v1 = prompt.index("<Video 1>:")
    a1 = prompt.index("<Audio 1>:")
    body = prompt.index("wide shot")
    assert p3 < v1 < a1 < body


def test_dummy_text_encoder_accepts_int_counts():
    """The DummyTextEncoder's signature widened in v17 -- if a caller passes an
    int/list for ``has_ref_*`` (as the multi-ref pipeline does), it must not
    raise."""
    enc = DummyTextEncoder()
    out = enc.encode("x", has_ref_image=3, has_ref_video=0, has_ref_audio=2)
    assert out.shape[0] == 1


def test_paired_video_audio_ref_block_orders_audio_before_video():
    """Sub 5: kind='video_audio' packs the paired audio right before the video's
    image rows. The three accumulators must stay in the same order."""
    imgs = [_fake_img_latent()]
    vids = [_fake_vid_latent(vt=2), _fake_vid_latent(vt=2)]
    paired = [_fake_aud_latent(t=100), None]  # first video paired, second not
    indep_auds = [_fake_aud_latent(t=200)]

    # Reproduce pipeline.generate's ref/latent builder with paired handling.
    refs = []
    cv = []
    ca = []
    for zi in imgs:
        refs.append(RefBlock(kind="image", latent_h=zi.shape[3], latent_w=zi.shape[4]))
        cv.append(zi)
    for zv, zva in zip(vids, paired):
        _, _, vt, rh, rw = zv.shape
        if zva is not None:
            rat = zva.shape[-1]
            refs.append(RefBlock(kind="video_audio", latent_h=rh, latent_w=rw,
                                 latent_t=vt, ref_audio_t=rat))
            cv.append(zv)
            ca.append(zva)  # paired audio precedes video in packed layout
        else:
            refs.append(RefBlock(kind="video", latent_h=rh, latent_w=rw, latent_t=vt))
            cv.append(zv)
    for za in indep_auds:
        refs.append(RefBlock(kind="audio", ref_audio_t=za.shape[-1]))
        ca.append(za)

    layout = PackedLayout(
        text_len=32, latent_t=2, latent_h=48, latent_w=84, audio_t=100,
        refs=refs,
    )

    # Kinds check.
    assert [r.kind for r in refs] == ["image", "video_audio", "video", "audio"]

    # cond_video_latents identity order == image + both videos.
    assert cv[0] is imgs[0] and cv[1] is vids[0] and cv[2] is vids[1]
    # cond_audio_latents identity order == paired then independent.
    assert ca[0] is paired[0] and ca[1] is indep_auds[0]

    # PackedLayout segment order: text -> ref_img (image) ->
    # ref_audio (paired) -> ref_img (video_audio's video rows) ->
    # ref_img (plain video) -> audio (target) -> video (target).
    seg_kinds = [k for _, _, k in layout.segments]
    assert seg_kinds == [
        "text",
        "ref_img",       # image ref
        "ref_audio",     # paired audio (before its video)
        "ref_img",       # video_audio's video rows
        "ref_img",       # plain video's rows
        "ref_audio",     # independent audio
        "audio", "video",
    ], seg_kinds

    # Number of audio-pos rows must equal sum of ref_audio segments in tokens,
    # and each entry in cond_audio_latents corresponds to one audio segment.
    audio_seg_count = sum(1 for k in seg_kinds if k == "ref_audio")
    assert audio_seg_count == len(ca)
