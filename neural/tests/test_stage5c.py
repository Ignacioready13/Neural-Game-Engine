"""Stage 5c unit tests: the new conditioning defaults, and the old models unchanged.

Three things matter and nothing else does (Stage 5c verification item 1):

(a) the new frame-conditioning features default to the clean, not-dropped case at
    inference (level 0), so a Stage 5c checkpoint behaves deterministically when the
    player does not ask for anything special;
(b) every *older* model still loads and predicts **bit-identically** — checked two
    ways: against golden digests captured from the pre-Stage-5c ``flow.py`` (fetched
    from git history before this stage's change; see ``_GOLDEN``), and by the existing
    replay round-trip test in ``test_stage6.py``, which the suite still runs;
(c) the Stage 5c helper live-path maths (``stage5c_rollout.Drive``) is the player's
    maths, so a decision rollout and a played rollout cannot drift apart.

CPU-only except where a model load is unavoidable, which uses the GPU when present.
"""

import hashlib
import os

import numpy as np
import pytest
import torch

import model_registry as registry
import stage5c_rollout as R
from flow import FlowSteerModel, flow_training_loss, load_flow_checkpoint
from steer import FEAT_DIM

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = os.path.abspath(os.path.join(HERE, "..", "models", "seeds.npz"))

# sha256 (first 24 hex) of predict(ctx, features=feats, seed=4242) with the fixed
# 4x3x48x64 context below, computed with the pre-Stage-5c flow.py.
_GOLDEN = {
    "flow96": "bb90878c3b367b8f2413295f",
    "flow96c": "522a331c282d43f6a7498efa",
}


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _fixed_inputs():
    torch.manual_seed(1234)
    return torch.randn(1, 4, 3, 48, 64), torch.randn(1, 4, FEAT_DIM)


def _digest(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:24]


# --------------------------------------------------------------------------- #
# (b) older checkpoints are unchanged
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["flow96", "flow96c"])
def test_old_checkpoint_predicts_bit_identically(name):
    ckpt, _ = registry.checkpoint_for(registry.get(name))
    model, payload = load_flow_checkpoint(ckpt, device="cpu")
    ctx, feats = _fixed_inputs()
    with torch.no_grad():
        out = model.predict(ctx, features=feats, seed=4242)
    assert _digest(out) == _GOLDEN[name]


def test_old_checkpoint_has_no_new_conditioning_heads():
    ckpt, _ = registry.checkpoint_for(registry.get("flow96c"))
    model, _ = load_flow_checkpoint(ckpt, device="cpu")
    assert model.level_embed is None and model.null_embed is None
    assert model.cond_dim == model.K * FEAT_DIM + model.H + 1


def test_new_kwargs_are_no_ops_for_an_old_checkpoint():
    ckpt, _ = registry.checkpoint_for(registry.get("flow96"))
    model, _ = load_flow_checkpoint(ckpt, device="cpu")
    ctx, feats = _fixed_inputs()
    with torch.no_grad():
        a = model.predict(ctx, features=feats, seed=7)
        b = model.predict(ctx, features=feats, seed=7, level=0, guidance=1.0)
    assert torch.equal(a, b)


def test_registered_models_still_load_and_replay_through_the_player(tmp_path):
    """The registry path the player uses, on the frozen copies, one full round trip."""
    import play
    bank = play.CacheSeeds(SEEDS)
    seed = bank.pick(np.random.default_rng(11), "val")
    ckpt, _ = registry.checkpoint_for(registry.get("flow96c"))
    player = play.Player(ckpt, bank, device=_device(), name="flow96c")
    clicks = [(3, 0.7), (11, -0.9)]
    player.reset(seed=seed)
    first = player.generate(12, clicks, noise_seed=5)
    rec = play.SessionRecorder("flow96c", ckpt, seed, 5)
    for cf, cx in clicks:
        rec.click(cf, cx)
    rec.advance(12)
    path = rec.save(str(tmp_path / "rt.json"))
    player.reset(seed=seed)
    again = player.generate(12, play.Replay(path).clicks, noise_seed=5)
    assert all(torch.equal(a, b) for a, b in zip(first, again))


# --------------------------------------------------------------------------- #
# (a) the new conditioning defaults to clean at inference
# --------------------------------------------------------------------------- #

def _new_model(**kw):
    cfg = dict(K=4, H=10, base=32, steps=4, cond_dim=128, net_base=16, attn=False,
               noise_level_max=0.5, level_buckets=10, ctx_dropout=0.1)
    cfg.update(kw)
    return FlowSteerModel(**cfg)


def test_level_defaults_to_zero_at_inference():
    model = _new_model()
    ctx, feats = _fixed_inputs()
    with torch.no_grad():
        default = model.predict(ctx, features=feats, seed=3)
        zero = model.predict(ctx, features=feats, seed=3, level=0)
        high = model.predict(ctx, features=feats, seed=3, level=10)
    assert torch.equal(default, zero)
    assert not torch.equal(default, high)


def test_level_conditioning_changes_the_condition_vector_only_through_its_embedding():
    model = _new_model()
    ctx, feats = _fixed_inputs()
    lv0 = torch.zeros(1, dtype=torch.long)
    lv5 = torch.full((1,), 5, dtype=torch.long)
    _, _, _, _, c0 = model.condition(ctx, feats, level=lv0)
    _, _, _, _, c5 = model.condition(ctx, feats, level=lv5)
    assert c0.shape == c5.shape
    assert c0.shape[1] == model.cond_dim
    assert not torch.equal(c0, c5)


def test_context_dropout_adds_a_null_token_and_guidance_only_when_asked():
    model = _new_model(noise_level_max=0.0, level_buckets=0)
    ctx, feats = _fixed_inputs()
    with torch.no_grad():
        plain = model.predict(ctx, features=feats, seed=3)
        guid = model.predict(ctx, features=feats, seed=3, guidance=1.5)
    assert plain.shape == guid.shape and not torch.equal(plain, guid)


def test_noise_level_corruption_makes_a_loss_that_backpropagates():
    model = _new_model()
    ctx, feats = _fixed_inputs()
    tgt = torch.randn(1, 3, 48, 64)
    loss, _ = flow_training_loss(model, ctx, feats, tgt, noise_level_max=0.5,
                                 level_buckets=10, ctx_dropout=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.net.parameters())
    # and the level-0 default at inference is what the player will use
    with torch.no_grad():
        model.predict(ctx, features=feats, seed=1)


def test_conditioning_is_off_by_default():
    model = FlowSteerModel(K=4, H=10, base=32, steps=4, cond_dim=128, net_base=16)
    assert model.level_embed is None and model.null_embed is None
    assert model.cond_dim == 4 * FEAT_DIM + 10 + 1


# --------------------------------------------------------------------------- #
# (c) the helper maths is the player's maths
# --------------------------------------------------------------------------- #

def test_drive_matches_the_player_bit_for_bit():
    import play
    bank = play.CacheSeeds(SEEDS)
    seed = bank.pick(np.random.default_rng(21), "val")
    ckpt, _ = registry.checkpoint_for(registry.get("flow96"))
    player = play.Player(ckpt, bank, device=_device(), name="flow96")
    clicks = [(5, 0.6), (17, -0.8)]
    played = R.rollout_live(player, seed, steps=12, clicks=clicks, noise_seed=9)
    drive = R.Drive(player.model, seed, device=player.device, K=player.K,
                    hist_extra=player.hist_extra).generate(
        12, clicks=clicks, noise_seed=9)
    assert np.array_equal(played, drive)


def test_noise_corr_zero_is_the_unchanged_default(tmp_path):
    import play
    bank = play.CacheSeeds(SEEDS)
    seed = bank.pick(np.random.default_rng(31), "val")
    ckpt, _ = registry.checkpoint_for(registry.get("flow96"))
    a = R.rollout_live(play.Player(ckpt, bank, device=_device()), seed, steps=10,
                       noise_seed=2)
    b = R.rollout_live(play.Player(ckpt, bank, device=_device(), noise_corr=0.0),
                       seed, steps=10, noise_seed=2)
    assert np.array_equal(a, b)


def test_noise_corr_is_deterministic_and_temporally_smoother():
    import play
    bank = play.CacheSeeds(SEEDS)
    seed = bank.pick(np.random.default_rng(41), "val")
    ckpt, _ = registry.checkpoint_for(registry.get("flow96"))
    p = play.Player(ckpt, bank, device=_device(), noise_corr=0.9)
    a = R.rollout_live(p, seed, steps=60, noise_seed=3)
    b = R.rollout_live(p, seed, steps=60, noise_seed=3)
    assert np.array_equal(a, b)
    fresh = R.rollout_live(play.Player(ckpt, bank, device=_device()), seed, steps=60,
                           noise_seed=3)
    assert not np.array_equal(a, fresh)
    assert R.health(a)["frame_change"] < R.health(fresh)["frame_change"]


# --------------------------------------------------------------------------- #
# The health numbers and the pass band
# --------------------------------------------------------------------------- #

def test_health_separates_a_static_rollout_from_a_changing_one():
    rng = np.random.default_rng(0)
    still = np.full((120, 16, 24, 3), 128, np.uint8)
    moving = np.stack([np.full((16, 24, 3), (40 + 8 * i) % 255, np.uint8)
                       for i in range(120)])
    noisy = np.clip(rng.normal(120, 60, (120, 16, 24, 3)), 0, 255).astype(np.uint8)
    assert R.health(still)["frame_change"] < 1e-6
    assert R.health(moving)["frame_change"] > R.health(still)["frame_change"]
    assert R.health(noisy)["hf_energy"] > R.health(still)["hf_energy"]


def test_pass_band_rejects_collapse_and_accepts_the_calibrated_rows():
    # the four Phase-0 rows (see STAGE5C-PROGRESS.md 4.0d)
    rows = {
        "real": dict(frame_change=4.60, hf_ratio=1.00, contrast=51.2, luma_mean=100.7,
                     near_black=0.0),
        "flow96": dict(frame_change=12.90, hf_ratio=1.068, contrast=44.7, luma_mean=128.6,
                       near_black=0.0),
        "flow96c": dict(frame_change=37.03, hf_ratio=2.696, contrast=38.3, luma_mean=178.9,
                        near_black=0.0),
        "flow96d": dict(frame_change=13.83, hf_ratio=1.672, contrast=21.4, luma_mean=172.0,
                        near_black=0.0),
        "frozen_noise": dict(frame_change=0.40, hf_ratio=2.73, contrast=78.1,
                             luma_mean=178.4, near_black=0.0),
    }
    assert R.in_band(rows["real"])["all"]
    assert R.in_band(rows["flow96"])["all"]
    assert not R.in_band(rows["flow96c"])["all"]
    assert not R.in_band(rows["flow96d"])["all"]
    assert not R.in_band(rows["frozen_noise"])["all"]


def test_contact_sheet_writes_a_grid(tmp_path):
    frames = np.zeros((30, 12, 16, 3), np.uint8)
    p = R.contact_sheet(frames, str(tmp_path / "s.png"), (1, 10, 20), scale=2)
    assert os.path.exists(p) and os.path.getsize(p) > 0
