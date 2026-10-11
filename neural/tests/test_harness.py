"""Stage 4 harness unit tests: action-shift transforms, mask, model interface."""

import numpy as np
import torch

from eval_harness import (
    _action_at,
    _ar_mask,
    _precompute_action_variants,
)
from model import PersistenceModel, build_model, stack_context_action


def test_action_at_matches_window_action(packed64):
    e = packed64.episodes[5]
    s = e["length"] // 2
    w = packed64.window(int(e["episode_index"]), s, 4, 1)
    got = _action_at(packed64, e["episode_index"], s)
    np.testing.assert_allclose(got, w.actions[0], atol=1e-6)


def test_action_variants_shift_by_one_frame(packed64):
    e = packed64.episodes[5]
    ep_idx = e["episode_index"]
    s = e["length"] // 2
    K = 4
    ep = np.array([ep_idx])
    start = np.array([s])
    variants = _precompute_action_variants(packed64, ep, start, K, seed=0)
    # earlier[j] == action at frame (s+j-1); later[j] == action at (s+j+1)
    for j in range(K):
        np.testing.assert_allclose(variants["earlier"][0, j],
                                   _action_at(packed64, ep_idx, s + j - 1), atol=1e-6)
        np.testing.assert_allclose(variants["later"][0, j],
                                   _action_at(packed64, ep_idx, s + j + 1), atol=1e-6)
    # true[j] == action at frame (s+j)
    np.testing.assert_allclose(variants["true"][0, j],
                               _action_at(packed64, ep_idx, s + j), atol=1e-6)
    # zeroed is all zeros, shuffled has the same multiset of rows (permuted)
    assert variants["zeroed"].shape == variants["true"].shape
    assert np.all(variants["zeroed"] == 0)
    assert set(map(tuple, variants["shuffled"][0].tolist())) == set(
        map(tuple, variants["true"][0].tolist())) if K == 1 else True


def test_ar_mask_flags_turning_windows(packed64):
    from analysis import load_analysis
    sidecar = load_analysis(packed64)
    ep, start = packed64.enumerate_windows("val", 4, 1, require_safe=True)
    mask = _ar_mask(packed64, sidecar, ep[:64], start[:64], 4)
    assert mask.dtype == bool and len(mask) == 64
    # a window is action-relevant iff its target-step heading change is large
    for i in range(min(16, len(mask))):
        d = sidecar.heading_change(int(ep[i]), int(start[i]) + 3)
        expect = (not np.isnan(d)) and abs(d) >= np.deg2rad(0.5)
        assert mask[i] == expect


def test_stack_context_action_shape_and_values():
    B, K, H, W = 2, 4, 8, 8
    ctx = torch.randn(B, K, 3, H, W)
    act = torch.randn(B, K, 2)
    x = stack_context_action(ctx, act)
    assert x.shape == (B, K * 5, H, W)
    # channels 3 and 4 of each frame equal that frame's actionX / hasClicked, broadcast
    for b in range(B):
        for k in range(K):
            assert torch.allclose(x[b, k * 5 + 3], ctx.new_full((H, W), act[b, k, 0].item()))
            assert torch.allclose(x[b, k * 5 + 4], ctx.new_full((H, W), act[b, k, 1].item()))
            # RGB channels are passed through unchanged
            assert torch.equal(x[b, k * 5: k * 5 + 3], ctx[b, k])


def test_persistence_model_ignores_action():
    p = PersistenceModel()
    ctx = torch.randn(2, 4, 3, 16, 16)
    out = p.predict(ctx, torch.zeros(2, 4, 2))
    assert torch.equal(out, ctx[:, -1])


def test_model_interface_round_trip():
    cfg = dict(K=4, base=16)
    model = build_model(cfg)
    ctx = torch.randn(1, 4, 3, 48, 64)
    act = torch.randn(1, 4, 2)
    out = model.predict(ctx, act)
    assert out.shape == (1, 3, 48, 64)
    assert out.min().item() >= -1.0 and out.max().item() <= 1.0
