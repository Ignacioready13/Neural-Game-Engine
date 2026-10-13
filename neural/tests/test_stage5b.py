"""Focused tests for Stage 5b additions (energy, judges, scorecard plumbing).

Run:  ./memcap.sh --mem 6G -- .venv/bin/python -m pytest tests/test_stage5b.py -q
"""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from powerlog import PowerLogger, integrate_wh, mean_of          # noqa: E402
from judges import (health_stats, opening_events, radial_psd, _auc,  # noqa: E402
                    spectrum_report, multiscale_gradient, to_float01)


# --------------------------------------------------------------------------- #
# energy
# --------------------------------------------------------------------------- #

def test_integrator_constant_power():
    t = list(range(3601))
    assert integrate_wh(t, [100.0] * 3601) == pytest.approx(100.0, abs=1e-9)


def test_integrator_ramp_and_short_window():
    t = list(range(3601))
    ramp = [200.0 * i / 3600 for i in t]
    assert integrate_wh(t, ramp) == pytest.approx(100.0, abs=1e-9)   # trapezoid of a ramp
    assert integrate_wh(list(range(31)), [45.0] * 31) == pytest.approx(0.375, abs=1e-9)


def test_integrator_degenerate_inputs():
    assert integrate_wh([], []) == 0.0
    assert integrate_wh([1.0], [5.0]) == 0.0
    assert integrate_wh([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]) == 0.0   # zero dt


def test_integrator_ignores_non_monotonic():
    # a repeated timestamp contributes nothing rather than a negative Wh
    assert integrate_wh([0.0, 10.0, 10.0, 20.0], [10.0, 10.0, 10.0, 10.0]) \
        == pytest.approx(20.0 * 10.0 / 3600.0, abs=1e-12)


def test_logger_records_a_run():
    lg = PowerLogger(interval=0.05).start()
    assert lg._thread is not None
    s = lg.stop()
    assert s["duration_s"] >= 0
    assert s["samples"] >= 1
    assert "gpu_wh" in s and "throttle_events" in s
    assert mean_of([None, 2.0, 4.0]) == 3.0
    assert mean_of([None]) is None


# --------------------------------------------------------------------------- #
# judges
# --------------------------------------------------------------------------- #

def _frames(n=6, h=24, w=32, seed=0, kind="noise"):
    rng = np.random.RandomState(seed)
    if kind == "noise":
        return (rng.rand(n, h, w, 3) * 255).astype(np.uint8)
    if kind == "blur":       # low frequency: a smooth gradient
        yy, xx = np.mgrid[0:h, 0:w]
        base = (yy / h + xx / w)[None, :, :, None].repeat(n, 0).repeat(3, 3)
        return (np.clip(base, 0, 1) * 255).astype(np.uint8)
    raise ValueError(kind)


def test_radial_psd_shape_and_grain_sensitivity():
    f, p, c = radial_psd(_frames(seed=1), n_bins=12)
    assert len(f) == len(p) == len(c) == 12
    assert f[0] == 0.0 and p.min() > 0
    grain = spectrum_report(_frames(seed=2), _frames(seed=3))
    blur = spectrum_report(_boxdown(_frames(seed=5)), _frames(seed=5))
    assert grain["spectrum_distance"] >= 0 and blur["spectrum_distance"] >= 0
    # blurring really does lose high-frequency power relative to the source
    assert blur["band_ratio"]["0.35-0.51"] < 1.0


def _boxdown(seq, k=4):
    """A blurred copy of the same content (downsample + nearest upsample)."""
    import torch.nn.functional as F
    t = torch.from_numpy(to_float01(seq)).permute(0, 3, 1, 2)
    b = F.avg_pool2d(t, k)
    b = F.interpolate(b, size=t.shape[-2:], mode="nearest")
    return (b.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)


def test_multiscale_gradient_drops_with_blur():
    sharp = multiscale_gradient(_frames(seed=4))
    smooth = multiscale_gradient(_boxdown(_frames(seed=4)))
    for k in ("grad_scale1", "grad_scale2", "grad_scale4"):
        assert sharp[k] > smooth[k], k


def test_auc_extremes_and_ties():
    assert _auc([1.0, 2.0, 3.0], [0.0, -1.0, -2.0]) == pytest.approx(1.0)
    assert _auc([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]) == pytest.approx(0.0)
    assert _auc([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.5)
    assert _auc([], [1.0]) == 0.5


def test_opening_events_counts_rising_edges():
    pos = np.zeros(40, dtype=bool)
    pos[10:14] = True          # a clean opening event
    pos[25] = True             # a one-frame blip after only 10 negatives: counts too
    ev = opening_events(pos)
    assert ev["n_events"] == 2
    assert ev["events_per_100"] == pytest.approx(2 / 40 * 100)
    assert ev["positive_share"] == pytest.approx(5 / 40)
    assert opening_events(np.zeros(0, dtype=bool))["n_events"] == 0


def test_health_stats_units_and_collapse_flags():
    seq = _frames(10, 16, 16, seed=7, kind="noise")
    st = health_stats(seq)
    assert st["mean_luminance"] > 0 and st["frame_change"] > 0
    assert st["frozen"] is False and st["collapse_black"] is False
    black = np.zeros((10, 16, 16, 3), dtype=np.uint8)
    stb = health_stats(black)
    assert stb["collapse_black"] is True and stb["frozen"] is True
    white = np.full((10, 16, 16, 3), 255, dtype=np.uint8)
    assert health_stats(white)["collapse_white"] is True


def test_health_dark_region_colour():
    seq = _frames(4, 16, 16, seed=8)
    seq[:] = 140
    seq[:, :8] = np.array([20, 6, 3], dtype=np.uint8)     # a red-ish dark band
    st = health_stats(seq)
    assert st["dark_pixels"] > 0
    assert st["dark_r_minus_b"] > 0.05                      # red above blue in the dark
    assert to_float01(seq).max() <= 1.0


# --------------------------------------------------------------------------- #
# the batched rollout must equal the single-rollout reference
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not torch.cuda.is_available() or not os.path.exists(
    os.path.join(HERE, "runs", "flow96", "best.pt")), reason="needs GPU + checkpoint")
def test_batched_rollout_matches_single():
    from dataset import PackedDataset
    from eval_harness import rollout_v2
    from gates import load_model
    import scorecard as SC

    sets = SC.load_sets()
    packed = PackedDataset(sets["packed"])
    K = sets["K"]
    model, _ = load_model(os.path.join(HERE, "runs", "flow96", "best.pt"), "cuda")
    clips = [(c["episode"], c["start"]) for c in sets["main_clips"][:2]]
    for scenario in ("no_click", "real_click"):
        for clip in clips:
            # batch of one: the sampler draws the same noise as rollout_v2, so the
            # batched path must reproduce it bit for bit
            override = torch.zeros(1, K, 2) if scenario == "no_click" else None
            gen, _ = rollout_v2(packed, model, clip[0], clip[1], K, 8,
                                action_override=override, seed=0, device="cuda")
            ref = SC._u8(gen)
            solo = SC._rollout_batch(packed, model, [clip], K, 8, scenario, 0, "cuda")[0]
            d = np.abs(ref.astype(np.int16) - SC._u8(solo).astype(np.int16))
            assert d.max() == 0, f"{scenario} batch=1: max diff {d.max()}"
        # With a batch > 1 the sampler's noise tensor has a different shape, so the
        # draws differ and two stochastic rollouts diverge; that is why every
        # candidate is scored with the same batch size.  Only batch=1 can be
        # expected to reproduce rollout_v2 exactly.


# --------------------------------------------------------------------------- #
# the revealed-strip weighting used by I4, and the no-oracle boundary
# --------------------------------------------------------------------------- #

def test_revealed_mask_shape_mean_and_sides():
    from flow import revealed_mask
    m = revealed_mask(torch.tensor([3.0, -2.0, 0.0]), 4, 8, edge_weight=6.0)
    assert tuple(m.shape) == (3, 1, 4, 8)
    # normalised: the batch mean is 1, so the loss scale does not change
    assert float(m.mean()) == pytest.approx(1.0, abs=1e-6)
    # a rightward content shift reveals the LEFT columns; leftward reveals the right
    assert float(m[0, 0, 0, 0]) > float(m[0, 0, 0, 6])
    assert float(m[1, 0, 0, 7]) > float(m[1, 0, 0, 1])
    # near-zero shift still weights a 3-column leading strip (`min_cols`): the strip
    # that is about to be revealed as soon as the shift goes positive, which keeps the
    # weighting continuous in dx instead of switching on at 1 px
    assert float(m[2, 0, 0, 0]) > float(m[2, 0, 0, 7])
    # edge_weight 0 disables it entirely
    assert float(revealed_mask(torch.tensor([5.0]), 4, 8, edge_weight=0.0).max()) == 1.0


def test_inference_path_does_not_import_the_judges():
    """The opening detector and the real-vs-fake classifier are evaluation-only."""
    import re
    root = HERE
    for name in ("steer.py", "flow.py", "play.py", "eval_harness.py", "dataset.py"):
        src = open(os.path.join(root, name)).read()
        for pat in (r"^\s*import\s+judges", r"^\s*from\s+judges\s+import",
                    r"^\s*import\s+scorecard", r"^\s*from\s+scorecard\s+import"):
            assert not re.search(pat, src, re.M), f"{name} imports a judge: {pat}"
    # and the judges module itself only uses the sidecar for labels, never for pixels
    jsrc = open(os.path.join(root, "judges.py")).read()
    assert "sidecar" in jsrc and "def opening_labels" in jsrc
