"""Stage 5 unit tests: warp, click features, mirror augmentation, metrics, no-oracle.

Small and CPU-only. Data-dependent tests use the Stage 2 pilot packed sets
(``packed64``/``packed96`` fixtures from conftest.py); nothing here needs a GPU.
"""

import inspect

import numpy as np
import pytest
import torch

import eval_harness
from metrics import black_frac, color_stats, frame_change, sharpness, seed_diversity
from steer import SINCE_CLIP, click_features, warp_translate


# --------------------------------------------------------------------------- #
# Warp / resampling
# --------------------------------------------------------------------------- #

def _img(h=48, w=64):
    rng = np.random.default_rng(0)
    return torch.from_numpy(rng.random((1, 3, h, w)).astype(np.float32)) * 2 - 1


def test_warp_identity_is_exact():
    x = _img()
    out = warp_translate(x, torch.zeros(1))
    assert torch.allclose(out, x, atol=1e-5)


@pytest.mark.parametrize("dx", [1.0, 2.0, -3.0])
def test_warp_integer_shift_moves_content(dx):
    from metrics import estimate_shift_x
    x = _img()
    out = warp_translate(x, torch.full((1,), float(dx)))
    a = ((x[0].permute(1, 2, 0) * 0.5 + 0.5) * 255).numpy().astype(np.uint8)
    b = ((out[0].permute(1, 2, 0) * 0.5 + 0.5) * 255).numpy().astype(np.uint8)
    assert abs(estimate_shift_x(a, b) - dx) < 0.5


@pytest.mark.parametrize("dx", [0.6, -0.6, 1.4, -1.4])
def test_warp_subpixel_shift_recovers_sign_and_magnitude(dx):
    from metrics import estimate_shift_x
    x = _img()
    out = warp_translate(x, torch.full((1,), float(dx)))
    a = ((x[0].permute(1, 2, 0) * 0.5 + 0.5) * 255).numpy().astype(np.uint8)
    b = ((out[0].permute(1, 2, 0) * 0.5 + 0.5) * 255).numpy().astype(np.uint8)
    d = estimate_shift_x(a, b)
    # the phase-correlation estimator quantises to the nearest pixel, so only the
    # sign and the rough magnitude are asserted here
    assert np.sign(d) == np.sign(dx)
    assert abs(d - dx) <= 1.0


def test_warp_border_pads_instead_of_blacking():
    x = _img()
    out = warp_translate(x, torch.full((1,), 4.0))
    # content moved right: the leftmost output columns repeat the left border
    assert torch.allclose(out[:, :, :, 0], x[:, :, :, 0], atol=1e-5)
    assert out.min() > -1.0 - 1e-5


# --------------------------------------------------------------------------- #
# Click features
# --------------------------------------------------------------------------- #

def test_click_features_hold_last_and_delta():
    acts = torch.zeros(1, 6, 2)
    acts[0, 2] = torch.tensor([0.5, 1.0])          # an accepted click at step 2
    acts[0, 3:] = torch.tensor([0.5, 1.0])         # held afterwards (the latch stays 1)
    f = click_features(acts)[0]
    # deltaActionX at the click = (0.5 - 0) / 2
    assert abs(f[2, 2].item() - 0.25) < 1e-6
    # held steps have zero delta
    assert abs(f[3, 2].item()) < 1e-6 and abs(f[4, 2].item()) < 1e-6
    # frames-since-click is 0 at the click and counts up afterwards
    assert abs(f[2, 3].item() - 0.0) < 1e-6
    assert abs(f[3, 3].item() - 1.0 / SINCE_CLIP) < 1e-6
    assert abs(f[4, 3].item() - 2.0 / SINCE_CLIP) < 1e-6
    # ... and the latch is not used for that (it stays 1 on every held step)
    assert float(f[4, 1]) == 1.0


def test_click_features_before_first_click_are_zero():
    acts = torch.zeros(1, 5, 2)
    f = click_features(acts)[0]
    assert abs(f[-1, 0].item()) < 1e-6 and abs(f[-1, 1].item()) < 1e-6


# --------------------------------------------------------------------------- #
# Mirror augmentation
# --------------------------------------------------------------------------- #

def test_mirror_augmentation_flips_and_negates(packed64):
    from train_steer import _batch_windows
    ep = np.array([30, 31], dtype=np.int64)
    start = np.array([60, 80], dtype=np.int64)
    cand = {"ep": ep, "start": start}
    cfg = {"K": 4, "mirror_prob": 0.0}
    ctx0, hist0 = _batch_windows(packed64, cand, np.array([0, 1]), cfg, "cpu",
                                 np.random.default_rng(0), extra_hist=4)
    cfg1 = {"K": 4, "mirror_prob": 1.0}     # rng.random() < 1.0 always True
    ctx1, hist1 = _batch_windows(packed64, cand, np.array([0, 1]), cfg1, "cpu",
                                 np.random.default_rng(0), extra_hist=4)
    # image flipped horizontally
    assert torch.allclose(ctx1, torch.flip(ctx0, dims=[3]), atol=1e-5)
    # actionX negated, actionHasClicked unchanged
    assert torch.allclose(hist1[:, :, 0], -hist0[:, :, 0], atol=1e-6)
    assert torch.allclose(hist1[:, :, 1], hist0[:, :, 1], atol=1e-6)


# --------------------------------------------------------------------------- #
# Stage 5 metrics
# --------------------------------------------------------------------------- #

def test_sharpness_drops_under_blur():
    x = torch.rand(1, 3, 48, 64) * 2 - 1
    k = torch.ones(1, 1, 5, 5) / 25.0
    blurred = torch.nn.functional.conv2d(x, k.repeat(3, 1, 1, 1), padding=2, groups=3)
    assert sharpness(blurred) < sharpness(x)


def test_color_stats_and_frame_change():
    x = torch.zeros(2, 3, 8, 8)
    cs = color_stats(x)
    assert cs["mean_all"] == pytest.approx(0.5)          # [-1,1] zero -> 0.5 in [0,1]
    assert frame_change(x, x) == 0.0
    assert frame_change(x, x + 0.2) == pytest.approx(0.1, abs=1e-6)
    assert black_frac(torch.full((1, 3, 8, 8), -1.0)) == 1.0


def test_seed_diversity_zero_when_identical():
    a = torch.rand(1, 3, 16, 16) * 2 - 1
    assert seed_diversity([a, a, a]) == 0.0
    b = a + 0.5
    assert seed_diversity([a, b]) > 0.0


# --------------------------------------------------------------------------- #
# No-oracle / N1-N2
# --------------------------------------------------------------------------- #

def test_inference_models_accept_only_frames_and_clicks():
    """The inference entry points take frames, click signals and a seed — nothing else."""
    import flow
    from steer import SteeringModel
    allowed = {"self", "context", "action", "features", "seed", "return_shift"}
    for cls in (SteeringModel, flow.FlowSteerModel):
        params = set(inspect.signature(cls.predict).parameters)
        assert params <= allowed, (cls.__name__, params - allowed)


def test_rollout_driver_never_reads_the_sidecar(packed64, monkeypatch):
    """rollout_v2 must work with the analysis sidecar unavailable (N1/N2)."""
    import analysis
    from eval_harness import rollout_v2
    from model import PersistenceModel

    def _boom(*a, **k):
        raise AssertionError("the rollout driver must not read the sidecar")

    monkeypatch.setattr(analysis, "load_analysis", _boom)
    model = PersistenceModel()
    frames, info = rollout_v2(packed64, model, 0, 30, 4, 3, device="cpu")
    assert frames.shape[0] == 3
    assert "analysis" not in inspect.getsource(rollout_v2)
    assert "sidecar" not in inspect.getsource(rollout_v2)


# --------------------------------------------------------------------------- #
# Player click semantics (engine: debounce 150 ms = 3 frames at 20 Hz)
# --------------------------------------------------------------------------- #

def test_player_click_debounce_and_clamp():
    from play import ClickState
    cs = ClickState()
    assert cs.has_clicked == 0.0
    assert cs.click(0.5, 0) is True                 # first click accepted
    assert cs.has_clicked == 1.0 and cs.x == 0.5
    assert cs.click(-0.5, 1) is False               # 1 frame later: ignored
    assert cs.x == 0.5
    assert cs.click(-0.5, 2) is False               # 2 frames later: ignored
    assert cs.click(-0.5, 3) is True                # 3 frames later: accepted
    assert cs.x == -0.5
    assert cs.click(5.0, 10) is True                # clamped
    assert cs.x == 1.0
    assert cs.click(-9.0, 20) is True
    assert cs.x == -1.0
