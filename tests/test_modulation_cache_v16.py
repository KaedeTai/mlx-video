"""v16 260807 dry-run: exact step-indexed cache + fail-closed drop.

Uses a synthetic mini-DiT (2 blocks, tiny hidden) so this runs in <1 s and
doesn't need the real 25 GiB checkpoint. Verifies:

  1. ModulationCache.build produces step-indexed tables.
  2. verify_cache_bitwise passes for a matched build/live pair.
  3. verify_cache_bitwise raises when the DiT weights change under the cache.
  4. drop_adaln_weights refuses on num_steps not in supported_set.
  5. drop_adaln_weights refuses on schedule_name mismatch.
  6. drop_adaln_weights refuses on use_adaln_curves=True.
  7. drop_adaln_weights refuses on signature-field mismatch.
  8. drop_adaln_weights succeeds when signature agrees, and blows the weights
     from adaln_proj.linear.
  9. The step-index gather is a pure list lookup (never touches float keys) --
     even a 30-step schedule with fp32 vs fp64 rounding drift cache-hits every
     step, unlike v15 which missed 2/30.

Run:  pytest tests/test_modulation_cache_v16.py -x -s
"""

from __future__ import annotations

import pytest

import mlx.core as mx
import mlx.nn as nn

from mlx_video.models.minimax_h3.blocks import AdalnProj, TimeEmbedder
from mlx_video.models.minimax_h3.modulation_cache import (
    DEFAULT_SCHEDULE_NAME,
    CacheSignature,
    DropRefused,
    ModulationCache,
    _hash_per_step_ut,
    _live_signature_from_pipeline,
    drop_adaln_weights,
    per_step_unique_t,
    verify_cache_bitwise,
)


# ---------------------------------------------------------------------------
# Synthetic mini-DiT with the two attributes the cache touches: time_embedder
# and blocks[].adaln_proj + final_layer.adaln_proj.
# ---------------------------------------------------------------------------


class _FakeBlock(nn.Module):
    def __init__(self, t_dim: int, hidden: int):
        super().__init__()
        self.adaln_proj = AdalnProj(t_dim, hidden, expand=6, modalities=3, apply_silu=True)


class _FakeFinal(nn.Module):
    def __init__(self, t_dim: int, hidden: int):
        super().__init__()
        self.adaln_proj = AdalnProj(t_dim, hidden, expand=2, modalities=1, apply_silu=True)


class _FakeDit(nn.Module):
    def __init__(self, n_blocks: int = 2, hidden: int = 32, t_dim: int = 24):
        super().__init__()
        self.time_embedder = TimeEmbedder(freq_dim=t_dim, hidden=t_dim * 2, out_dim=t_dim)
        self.blocks = [_FakeBlock(t_dim, hidden) for _ in range(n_blocks)]
        self.final_layer = _FakeFinal(t_dim, hidden)
        self.use_adaln_curves = False


def _make_sigmas(num_steps: int):
    """Cheap flow-matching-like schedule for the test."""
    import numpy as np
    from mlx_video.models.minimax_h3.scheduler import MiniMaxH3Scheduler
    sch = MiniMaxH3Scheduler()
    sch.set_timesteps(num_steps)
    return sch.sigmas.tolist(), sch


def _signature(sig_over=None, *, dit, sigmas_list, num_steps, has_vis, has_aud, sch):
    step_ut = per_step_unique_t(
        sigmas_list,
        has_visual_cond=has_vis, has_audio_cond=has_aud,
        shift_video=sch.shift_video, shift_audio=sch.shift_audio,
    )
    sig = _live_signature_from_pipeline(
        num_steps=num_steps, num_blocks=len(dit.blocks),
        has_visual_cond=has_vis, has_audio_cond=has_aud,
        shift_video=sch.shift_video, shift_audio=sch.shift_audio,
        sigma_min=1e-5, schedule_name=DEFAULT_SCHEDULE_NAME,
        dtype="float32", use_adaln_curves=False, cache_final=True,
        model_hash="test_model_hash", lora_hash=None,
        layout_hash="test_layout_hash",
        per_step_ut=step_ut, tag="test",
    )
    if sig_over:
        sig = CacheSignature(**{**sig.to_dict(), **sig_over})
    return sig, step_ut


def test_build_and_verify_matches():
    dit = _FakeDit()
    sigmas_list, sch = _make_sigmas(30)
    sig, step_ut = _signature(
        dit=dit, sigmas_list=sigmas_list, num_steps=30,
        has_vis=True, has_aud=True, sch=sch,
    )
    cache = ModulationCache.build(dit, step_ut, sig)
    n_checks, max_diff = verify_cache_bitwise(cache, dit, verbose=False)
    assert n_checks > 0
    assert max_diff == 0.0
    # 30 steps built, 2 blocks
    assert cache.num_steps == 30
    assert cache.num_blocks == 2
    # Step-index lookup is a list read -- no KeyError even on weird step counts.
    for i in range(30):
        tup = cache.gather(0, i)
        assert len(tup) == 6
        assert cache.final_layer_gather(i)


def test_verify_detects_weight_change():
    dit = _FakeDit()
    sigmas_list, sch = _make_sigmas(30)
    sig, step_ut = _signature(
        dit=dit, sigmas_list=sigmas_list, num_steps=30,
        has_vis=False, has_aud=False, sch=sch,
    )
    cache = ModulationCache.build(dit, step_ut, sig)
    # Perturb the DiT's adaln_proj weights so the cache no longer matches live.
    dit.blocks[0].adaln_proj.linear.weight = (
        dit.blocks[0].adaln_proj.linear.weight + 1e-3
    )
    with pytest.raises(RuntimeError, match="BITWISE MISMATCH"):
        verify_cache_bitwise(cache, dit, verbose=False)


def test_drop_refuses_bad_num_steps():
    dit = _FakeDit()
    sigmas_list, sch = _make_sigmas(30)
    sig, step_ut = _signature(
        dit=dit, sigmas_list=sigmas_list, num_steps=30,
        has_vis=False, has_aud=False, sch=sch,
    )
    cache = ModulationCache.build(dit, step_ut, sig)
    # 30 is normally supported; force a "weird" count into the signature and try.
    weird_sig = CacheSignature(**{**sig.to_dict(), "num_steps": 33})
    cache_weird = ModulationCache(
        per_step_tables=cache.per_step_tables,
        per_step_ut=cache.per_step_ut,
        per_step_final=cache.per_step_final,
        signature=weird_sig,
    )
    with pytest.raises(DropRefused, match="num_steps=33 not in supported set"):
        drop_adaln_weights(dit, cache_weird, live_signature=weird_sig, verbose=False)


def test_drop_refuses_bad_schedule():
    dit = _FakeDit()
    sigmas_list, sch = _make_sigmas(30)
    sig, step_ut = _signature(
        sig_over={"schedule_name": "karras_v1"},
        dit=dit, sigmas_list=sigmas_list, num_steps=30,
        has_vis=False, has_aud=False, sch=sch,
    )
    cache = ModulationCache.build(dit, step_ut, sig)
    with pytest.raises(DropRefused, match="schedule_name"):
        drop_adaln_weights(dit, cache, live_signature=sig, verbose=False)


def test_drop_refuses_curves_enabled():
    dit = _FakeDit()
    sigmas_list, sch = _make_sigmas(30)
    sig, step_ut = _signature(
        sig_over={"use_adaln_curves": True},
        dit=dit, sigmas_list=sigmas_list, num_steps=30,
        has_vis=False, has_aud=False, sch=sch,
    )
    cache = ModulationCache.build(dit, step_ut, sig)
    with pytest.raises(DropRefused, match="use_adaln_curves"):
        drop_adaln_weights(dit, cache, live_signature=sig, verbose=False)


def test_drop_refuses_signature_mismatch():
    dit = _FakeDit()
    sigmas_list, sch = _make_sigmas(30)
    sig, step_ut = _signature(
        dit=dit, sigmas_list=sigmas_list, num_steps=30,
        has_vis=False, has_aud=False, sch=sch,
    )
    cache = ModulationCache.build(dit, step_ut, sig)
    bad_live = CacheSignature(**{**sig.to_dict(), "has_visual_cond": True})
    with pytest.raises(DropRefused, match="signature mismatch"):
        drop_adaln_weights(dit, cache, live_signature=bad_live, verbose=False)


def test_drop_succeeds_and_frees_weights():
    dit = _FakeDit()
    sigmas_list, sch = _make_sigmas(30)
    sig, step_ut = _signature(
        dit=dit, sigmas_list=sigmas_list, num_steps=30,
        has_vis=False, has_aud=False, sch=sch,
    )
    cache = ModulationCache.build(dit, step_ut, sig)
    assert hasattr(dit.blocks[0].adaln_proj.linear, "weight")
    freed = drop_adaln_weights(
        dit, cache, live_signature=sig, verify_diff=0.0, verbose=False,
    )
    assert freed > 0
    assert not hasattr(dit.blocks[0].adaln_proj.linear, "weight")
    assert not hasattr(dit.final_layer.adaln_proj.linear, "weight")


def test_step_index_lookup_never_key_errors_at_odd_step_counts():
    """v16 fix: 30-step / 50-step / 100-step should all cache-hit every step."""
    for n_steps in (4, 30, 50, 100):
        dit = _FakeDit()
        sigmas_list, sch = _make_sigmas(n_steps)
        sig, step_ut = _signature(
            dit=dit, sigmas_list=sigmas_list, num_steps=n_steps,
            has_vis=True, has_aud=True, sch=sch,
        )
        cache = ModulationCache.build(dit, step_ut, sig)
        # Simulate the DiT forward loop touching every step index.
        for i in range(n_steps):
            tup = cache.gather(0, i)
            assert isinstance(tup, tuple) and len(tup) == 6
            final = cache.final_layer_gather(i)
            assert len(final) == 2
        with pytest.raises(IndexError):
            cache.gather(0, n_steps)
        with pytest.raises(IndexError):
            cache.gather(0, -1)


def test_signature_serialisation_roundtrip():
    """Signatures must round-trip through JSON so they can live in the stripped bundle."""
    import json
    dit = _FakeDit()
    sigmas_list, sch = _make_sigmas(30)
    sig, _ = _signature(
        dit=dit, sigmas_list=sigmas_list, num_steps=30,
        has_vis=False, has_aud=False, sch=sch,
    )
    blob = json.dumps(sig.to_dict())
    restored = CacheSignature.from_dict(json.loads(blob))
    ok, bad = sig.matches(restored)
    assert ok, bad


def test_hash_per_step_ut_stable():
    """Same input → same hash. Different input → different hash."""
    ut1 = [[0.1, 0.5], [0.2, 0.4]]
    ut2 = [[0.1, 0.5], [0.2, 0.4]]
    ut3 = [[0.1, 0.5], [0.2, 0.5]]
    assert _hash_per_step_ut(ut1) == _hash_per_step_ut(ut2)
    assert _hash_per_step_ut(ut1) != _hash_per_step_ut(ut3)
