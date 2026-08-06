"""Live-vs-cached AdaLN modulation parity for MiniMax-H3 (v15 260807 bugfix #3).

The v15 v14/v15 A/B was NOT controlled: different audio ref lengths (87 vs 116
audio latents) meant a different packed layout was fed into the DiT, so any
divergence could just as easily be from geometry as from the modulation cache.
This test isolates the cache path by holding EVERYTHING else fixed:

  * same seed, same prompt, same reference image, same reference audio wav
  * same scheduler / sigma schedule
  * same number of denoise steps
  * same DiT weights (only difference: cache built + weights dropped, or not)

and comparing the produced video/audio latents step-by-step for bitwise
equality.

Why this catches the "M=1 build" bug ChatGPT flagged
----------------------------------------------------
``ModulationCache.build`` iterates ``for i in range(T_all): t_single =
timesteps[i:i+1]`` -- always calling ``adaln_proj`` with M=1. The live path
calls it with M=len(unique_t) per step (typically 2-4). If MLX's Metal GEMM
kernel differs between M=1 and M=k, the cached rows won't match live rows.
A step-by-step latent hash compare will surface the discrepancy immediately,
whereas the previous single-point-check ``max_diff < 1e-5`` verify would miss
either (a) a single-t point that happened to match or (b) a systematic
kernel-order difference that lands inside the tolerance.

Run: ``pytest tests/test_modulation_cache_parity.py -x -s`` (needs 30-40 GB
free -- skip if the machine is under memory pressure).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("H3_PARITY", "0") != "1",
    reason="Set H3_PARITY=1 to run the H3 live-vs-cached parity test (loads DiT+VAE, "
           "~30-40 GB peak). Guarded off in default test runs.",
)


def _latent_hash(arr) -> str:
    import mlx.core as mx
    a = np.asarray(arr.astype(mx.float32)).tobytes()
    return hashlib.sha256(a).hexdigest()[:16]


def _first_100(arr) -> list:
    import mlx.core as mx
    return np.asarray(arr.astype(mx.float32)).ravel()[:100].tolist()


@pytest.fixture(scope="module")
def fixture_paths():
    """Fixed inputs: single wav, single ref image, single prompt.

    Kept minimal (320x480, L33, 4 steps) so the test runs in a couple of
    minutes on a busy machine. The specific images / wavs are not committed
    -- CI will skip via ``H3_PARITY`` gate.
    """
    root = Path(os.environ.get("H3_PARITY_ROOT", "~/tmp/h3_parity")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return {
        "ref_image": root / "ref_image.jpg",
        "ref_audio": root / "ref_audio.wav",
        "prompt": "王博士溫暖地看著鏡頭",
        "seed": 0,
        "width": 320,
        "height": 480,
        "length": 33,
        "num_steps": 4,
    }


def _run_once(cfg: dict, *, use_cache: bool):
    """Run the DiT denoise loop once, returning per-step (video_latent, audio_latent)
    hashes and the final generate() ``info`` dict.

    Instrumented copy of ``H3Pipeline.generate`` -- collects a snapshot after each
    scheduler step BEFORE VAE decode. That lets us compare exactly the tensors
    the DiT emits, isolating the adaln-cache change.
    """
    import mlx.core as mx
    from mlx_video.models.minimax_h3.pipeline import load_pipeline, temporal_shape
    from mlx_video.models.minimax_h3.packed_layout import PackedLayout, RefBlock

    pipe = load_pipeline()

    # Turbo LoRA path unused: keep the test surface small.
    if use_cache:
        pipe.build_adaln_cache_and_drop(
            num_steps=cfg["num_steps"],
            has_visual_cond=True,
            has_audio_cond=True,
            verbose=True,
        )

    # ---- Build refs identically for both paths ----
    from PIL import Image
    img = Image.open(cfg["ref_image"]).convert("RGB").resize(
        (cfg["width"], cfg["height"]), Image.LANCZOS)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 * 2.0 - 1.0)
    arr = arr.transpose(2, 0, 1)[None, :, None, :, :]
    ref_image_latent = pipe.video_vae.encode(mx.array(arr))

    from scipy.io import wavfile
    sr, wav = wavfile.read(cfg["ref_audio"])
    if wav.ndim == 1:
        wav = np.stack([wav, wav], axis=-1)
    wav = wav.astype(np.float32) / 32768.0
    ref_audio_latent = pipe.audio_vae.encode(mx.array(wav.T[None, ...]))

    # ---- Denoise loop copy (instrumented) ----
    frame_count, latent_t, audio_t = temporal_shape(cfg["length"])
    rng = np.random.default_rng(cfg["seed"])
    v = mx.array(rng.standard_normal(
        (1, 24, latent_t, cfg["height"] // 16, cfg["width"] // 16)
    ).astype(np.float32))
    a = mx.array(rng.standard_normal((1, 32, 2, audio_t)).astype(np.float32))

    ctx = pipe.text_encoder.encode(cfg["prompt"]).astype(mx.float32)
    text_len = ctx.shape[1]

    pipe.scheduler.set_timesteps(cfg["num_steps"])
    v = pipe.scheduler.scale_noise(v)
    a = pipe.scheduler.scale_noise(a)

    refs = [
        RefBlock(kind="image", latent_h=ref_image_latent.shape[-2],
                 latent_w=ref_image_latent.shape[-1]),
        RefBlock(kind="ref_audio", ref_audio_t=ref_audio_latent.shape[-1]),
    ]
    layout = PackedLayout(text_len, latent_t, cfg["height"] // 16,
                          cfg["width"] // 16, audio_t, refs=refs)
    payload = dict(
        refs=refs, layout=layout,
        cond_video_latents=[ref_image_latent],
        cond_audio_latents=[ref_audio_latent],
    )

    step_snaps = []
    for i in range(cfg["num_steps"]):
        ts = pipe.scheduler.timestep_for(i)
        v_v, v_a = pipe.dit((v, a), ts, ctx, payload=payload)
        mx.eval(v_v, v_a)
        v, a = pipe.scheduler.step(v_v, v_a, i, v, a)
        mx.eval(v, a)
        step_snaps.append({
            "step": i,
            "video_hash": _latent_hash(v),
            "audio_hash": _latent_hash(a),
            "video_first100": _first_100(v),
            "audio_first100": _first_100(a),
        })

    return step_snaps


def test_live_vs_cached_bitwise(fixture_paths):
    """Live and cached AdaLN paths must produce IDENTICAL per-step latents.

    Any mismatch here means the cache is not a valid drop-in replacement --
    either the build strategy diverges from the live GEMM path or the gather
    order is wrong.
    """
    if not fixture_paths["ref_image"].exists():
        pytest.skip(f"missing {fixture_paths['ref_image']}")
    if not fixture_paths["ref_audio"].exists():
        pytest.skip(f"missing {fixture_paths['ref_audio']}")

    import mlx.core as mx

    # Two independent DiT loads (avoid sharing lazy state)
    live = _run_once(fixture_paths, use_cache=False)
    try:
        mx.clear_cache()
    except AttributeError:
        pass
    cached = _run_once(fixture_paths, use_cache=True)

    assert len(live) == len(cached)
    diffs = []
    for lv, cv in zip(live, cached):
        if lv["video_hash"] != cv["video_hash"] or lv["audio_hash"] != cv["audio_hash"]:
            v_l = np.asarray(lv["video_first100"])
            v_c = np.asarray(cv["video_first100"])
            a_l = np.asarray(lv["audio_first100"])
            a_c = np.asarray(cv["audio_first100"])
            diffs.append({
                "step": lv["step"],
                "video_max": float(np.max(np.abs(v_l - v_c))),
                "audio_max": float(np.max(np.abs(a_l - a_c))),
                "live_video_hash": lv["video_hash"],
                "cached_video_hash": cv["video_hash"],
            })
    assert not diffs, (
        f"[cache parity] per-step latent divergence on {len(diffs)}/{len(live)} steps: "
        f"{diffs}"
    )
