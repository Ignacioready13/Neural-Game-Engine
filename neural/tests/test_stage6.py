"""Stage 6 unit tests: the player's geometry, click semantics, registry, replay.

Everything here is CPU-only except the two model-loading/replay checks, which use the
GPU when one is present and fall back to CPU. The window tests force SDL's dummy video
driver so they never open a real window.
"""

import os
import sys

import numpy as np
import pytest
import torch

import model_registry as registry
import play
from play import ClickState, image_rect, click_to_signal, signal_to_image_x, pulse_radius_px

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = os.path.join(HERE, "..", "models", "seeds.npz")
SEEDS = os.path.abspath(SEEDS)


# --------------------------------------------------------------------------- #
# Click mapping: letterbox + clamping, relative to the image
# --------------------------------------------------------------------------- #

def test_image_rect_letterboxes_and_centres():
    # a 1000x500 window for a 4:3 image: integer scale 6 -> 576x432, centred
    left, top, w, h, scale = image_rect(1000, 500, 96, 72, integer=True)
    assert (w, h) == (576, 432)
    assert (left, top) == (212, 34)
    assert scale == 6


def test_image_rect_uses_fractional_downscale_below_one():
    left, top, w, h, scale = image_rect(48, 36, 96, 72, integer=True)
    assert scale == pytest.approx(0.5)
    assert (w, h) == (48, 36)
    assert (left, top) == (0, 0)


def test_click_maps_relative_to_the_image_not_the_window():
    left, w = 212, 576                      # from the letterbox above
    assert click_to_signal(left, left, w) == pytest.approx(-1.0)          # image left edge
    assert click_to_signal(left + w, left, w) == pytest.approx(1.0)       # image right edge
    assert click_to_signal(left + w / 2, left, w) == pytest.approx(0.0)   # image centre
    assert click_to_signal(left + w / 4, left, w) == pytest.approx(-0.5)
    # clicks outside the image (the letterbox margins) clamp to the edges
    assert click_to_signal(0, left, w) == pytest.approx(-1.0)
    assert click_to_signal(999, left, w) == pytest.approx(1.0)
    assert click_to_signal(-50, left, w) == pytest.approx(-1.0)


def test_signal_to_image_x_inverts_the_mapping():
    left, w = 212, 576
    for x in (-1.0, -0.5, 0.0, 0.25, 1.0):
        back = (signal_to_image_x(x, left, w) - left) / w * 2 - 1
        assert back == pytest.approx(x)


def test_pulse_radius_matches_the_engine_and_grows_as_it_fades():
    w = 320.0                                # the engine's canvas width
    assert pulse_radius_px(w, 1.0) == pytest.approx(0.055 * w / 2)        # fresh
    assert pulse_radius_px(w, 0.0) == pytest.approx(0.1375 * w / 2)       # gone
    radii = [pulse_radius_px(w, life) for life in (1.0, 0.75, 0.5, 0.25, 0.0)]
    assert all(b > a for a, b in zip(radii, radii[1:]))


# --------------------------------------------------------------------------- #
# Click semantics: the engine's debounce, latch, and history reconstruction
# --------------------------------------------------------------------------- #

def test_player_click_debounce_and_clamp():
    cs = ClickState()
    assert cs.has_clicked == 0.0
    assert cs.click(0.5, 0) is True                 # first click accepted
    assert cs.has_clicked == 1.0 and cs.x == 0.5
    assert cs.click(-0.5, 1) is False               # 1 frame later: ignored
    assert cs.x == 0.5
    assert cs.click(-0.5, 2) is False               # 2 frames later: ignored
    assert cs.x == 0.5
    assert cs.click(-0.5, 3) is True                # 3 frames later: accepted
    assert cs.x == -0.5
    assert cs.click(5.0, 10) is True                # clamped
    assert cs.x == 1.0
    assert cs.click(-9.0, 20) is True
    assert cs.x == -1.0


def test_frames_since_accept_and_latch():
    cs = ClickState()
    assert cs.frames_since_accept(0) is None
    cs.click(0.2, 4)
    assert cs.frames_since_accept(4) == 0
    assert cs.frames_since_accept(11) == 7
    assert cs.has_clicked == 1.0                    # the latch never clears


def test_restore_reconstructs_the_debounce_from_a_real_history():
    # 12 rows wrapping the K=4 context: a click landed 2 frames before the window end
    hist = np.zeros((12, 2), np.float32)
    hist[9:, 0] = 0.6                               # held from row 9 on (offset -2)
    hist[9:, 1] = 1.0
    cs = ClickState()
    cs.restore(hist)
    assert cs.x == pytest.approx(0.6)
    assert cs.has_clicked == 1.0
    assert cs.frames_since_accept(0) == 2
    assert cs.click(-1.0, 0) is False               # only 2 frames since that accepted click
    assert cs.click(-1.0, 1) is True                # 3 frames later: accepted
    assert cs.x == pytest.approx(-1.0)


def test_restore_with_no_recent_change_leaves_the_first_click_accepted():
    hist = np.zeros((12, 2), np.float32)
    hist[:, 0] = 0.3                                # held since before the window
    hist[:, 1] = 1.0
    cs = ClickState()
    cs.restore(hist)
    assert cs.x == pytest.approx(0.3)               # the held value is kept
    assert cs.has_clicked == 1.0
    assert cs.click(-0.4, 0) is True                # no accepted click inside the window
    assert cs.x == pytest.approx(-0.4)


def test_restore_on_a_never_clicked_history():
    cs = ClickState()
    cs.restore(np.zeros((12, 2), np.float32))
    assert cs.x == 0.0 and cs.has_clicked == 0.0
    assert cs.frames_since_accept(0) is None
    assert cs.click(0.9, 0) is True                 # the very first click of the session


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

def test_registry_is_named_after_the_internal_runs():
    reg = registry.load()
    names = registry.names(reg)
    # the primary names are the internal training-run names, not 5a/5b
    assert "flow96" in names and "flow96c" in names
    assert registry.newest(reg)["name"] == "flow96c"
    assert registry.get("flow96", reg)["parent"] is None
    assert registry.get("flow96c", reg)["parent"] == "flow96"
    for name in ("flow96", "flow96c"):
        entry = registry.get(name, reg)
        assert registry.resolution(entry) == (96, 72)
        path, kind = registry.checkpoint_for(entry)
        assert kind == "copy"                       # frozen, so a rerun cannot move the name
        assert os.path.exists(path)


def test_unknown_model_reports_the_known_ones():
    with pytest.raises(registry.RegistryError) as e:
        registry.get("nope")
    text = str(e.value)
    assert "flow96" in text and "flow96c" in text


def test_aliases_resolve_to_the_internal_name():
    reg = registry.load()
    assert registry.canonical("5a", reg) == "flow96"
    assert registry.canonical("5b", reg) == "flow96c"
    assert registry.get("5a", reg)["name"] == "flow96"
    assert "5a" in registry.get("flow96", reg)["aliases"]


def test_selected_models_are_the_kept_revisions():
    """flow96d was rejected by the frozen scorecard, so it is not in the M cycle."""
    reg = registry.load()
    selected = registry.names(reg, selected_only=True)
    assert selected == ["flow96", "flow96c"]
    assert "flow96d" in registry.names(reg)
    assert registry.find("flow96d", reg)["selected"] is False
    assert registry.newest(reg)["name"] == "flow96c"


def test_hidden_flag_from_the_old_schema_reads_as_unselected(tmp_path):
    """A pre-Stage-6 registry used `hidden`; loading it must not break selection."""
    import json
    path = str(tmp_path / "registry.json")
    with open(path, "w") as fh:
        json.dump({"version": 1, "models": [
            {"name": "flow96", "hidden": False},
            {"name": "flow96d", "hidden": True}]}, fh)
    reg = registry.load(path)
    assert registry.names(reg, selected_only=True) == ["flow96"]
    assert registry.find("flow96d", reg)["selected"] is False


@pytest.mark.parametrize("bad", ["5", "A", "5A", "5!", "", "-5a", "5a b", "Flow96"])
def test_registry_rejects_bad_names(bad, tmp_path):
    # a tmp registry path: the test must never touch the real models/registry.json
    path = str(tmp_path / "registry.json")
    with pytest.raises(registry.RegistryError):
        registry.register(bad, "models/flow96/generator.pt", copy=False, path=path)
    assert not os.path.exists(path) or registry.load(path)["models"] == []


def test_registry_register_and_replace_roundtrip(tmp_path):
    path = str(tmp_path / "registry.json")
    src = registry.resolve("models/flow96/generator.pt")
    entry = registry.register("flow128", src, description="temp", width=96, height=72,
                              aliases=["6a"], copy=False, path=path)
    assert entry["name"] == "flow128" and entry["aliases"] == ["6a"]
    loaded = registry.load(path)
    assert registry.names(loaded) == ["flow128"]
    assert registry.canonical("6a", loaded) == "flow128"
    assert registry.names(loaded, selected_only=True) == ["flow128"]

    # a not-selected candidate is registered but stays out of the cycle
    registry.register("flow128b", src, description="candidate", copy=False,
                      selected=False, path=path)
    loaded = registry.load(path)
    assert registry.names(loaded) == ["flow128", "flow128b"]
    assert registry.names(loaded, selected_only=True) == ["flow128"]

    redone = registry.register("flow128", src, description="temp again", copy=False,
                               force=True, path=path)
    assert redone["description"] == "temp again"
    assert registry.names(registry.load(path)) == ["flow128b", "flow128"]
    with pytest.raises(registry.RegistryError):
        registry.register("flow128", src, copy=False, path=path)   # already there


def test_registering_a_duplicate_alias_needs_force(tmp_path):
    path = str(tmp_path / "registry.json")
    src = registry.resolve("models/flow96/generator.pt")
    registry.register("flow96", src, aliases=["5a"], copy=False, path=path)
    with pytest.raises(registry.RegistryError):
        registry.register("flow200", src, aliases=["5a"], copy=False, path=path)
    registry.register("flow200", src, aliases=["5a"], copy=False, force=True, path=path)
    reg = registry.load(path)
    assert registry.find("5a", reg)["name"] == "flow200"
    assert registry.find("flow96", reg)["aliases"] == []


# --------------------------------------------------------------------------- #
# Model loading and the closed loop
# --------------------------------------------------------------------------- #

def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("name", ["flow96", "flow96c", "flow96d"])
def test_both_registered_models_load_and_predict(name):
    from play import PackedSeeds, Player
    ckpt, _ = registry.checkpoint_for(registry.get(name))
    bank = _seed_bank()
    player = Player(ckpt, bank, device=_device(), name=name)
    frame = player.reset(seed=bank.pick(np.random.default_rng(0), "val"))
    assert player.K == 4
    out = player.step(player.clicks.action, seed=0)
    assert out.shape == (3, bank.height, bank.width)
    assert float(out.min()) >= -1.0 and float(out.max()) <= 1.0


def _seed_bank():
    if not os.path.exists(SEEDS):
        pytest.skip("no seed cache; run build_seeds.py")
    return play.CacheSeeds(SEEDS)


def test_step_feeds_only_frames_clicks_and_a_seed(monkeypatch):
    """N1/N2: the model call carries the frame context, the click signals and a seed."""
    from play import Player
    bank = _seed_bank()
    ckpt, _ = registry.checkpoint_for(registry.get("flow96"))
    player = Player(ckpt, bank, device=_device())
    seen = {}
    real = play.model_predict

    def spy(model, context, action=None, features=None, seed=None, return_shift=False):
        seen.update(context=context, action=action, features=features, seed=seed)
        return real(model, context, action, features, seed, return_shift)

    monkeypatch.setattr(play, "model_predict", spy)
    player.reset(seed=bank.pick(np.random.default_rng(1), "val"))
    player.step(player.clicks.action, seed=7)
    assert seen["context"].shape[:2] == (1, player.K)
    assert seen["action"].shape == (1, player.K, 2)
    assert seen["features"].shape[:2] == (1, player.K)
    assert seen["seed"] == 7
    # the action tensor is the real click history, not the held action repeated
    assert torch.allclose(seen["action"][0, -1].cpu(),
                          torch.tensor([player.clicks.x, player.clicks.has_clicked]))


def test_history_stays_bounded_over_a_long_session():
    """The feature loop is O(history); the history must not grow without bound."""
    from play import Player
    bank = _seed_bank()
    ckpt, _ = registry.checkpoint_for(registry.get("flow96"))
    player = Player(ckpt, bank, device=_device(), hist_extra=8)
    player.reset(seed=bank.pick(np.random.default_rng(2), "val"))
    for i in range(12):
        player.step((0.1 * (i % 3) - 0.1, 1.0), seed=i)
    assert len(player.hist) == player.hist_extra + player.K


def test_player_never_reads_the_sidecar(monkeypatch):
    """N1/N2 for the player: a rollout works with the analysis sidecar unavailable."""
    import analysis
    from play import Player

    def _boom(*a, **k):
        raise AssertionError("the player must not read the sidecar")

    monkeypatch.setattr(analysis, "load_analysis", _boom)
    bank = _seed_bank()
    ckpt, _ = registry.checkpoint_for(registry.get("flow96"))
    player = Player(ckpt, bank, device=_device())
    player.reset(seed=bank.pick(np.random.default_rng(3), "val"))
    for i in range(3):
        player.step(player.clicks.action, seed=i)


# --------------------------------------------------------------------------- #
# Recording and replay
# --------------------------------------------------------------------------- #

def test_record_and_replay_is_bit_identical(tmp_path):
    """Replaying a recording on the same model reproduces the frames exactly."""
    from play import Player, SessionRecorder, Replay
    bank = _seed_bank()
    seed = bank.pick(np.random.default_rng(4), "val")
    clicks = [(3, 0.7), (11, -0.9), (19, 0.4)]
    ckpt, _ = registry.checkpoint_for(registry.get("flow96c"))
    player = Player(ckpt, bank, device=_device(), name="flow96c")
    player.reset(seed=seed)
    first = player.generate(24, clicks, noise_seed=5)

    rec = SessionRecorder("flow96c", ckpt, seed, 5)
    for f, x in clicks:
        rec.click(f, x)
    rec.advance(24)
    path = rec.save(str(tmp_path / "session.json"))
    rep = Replay(path)
    assert rep.n_frames == 24 and rep.noise_seed == 5
    assert rep.clicks_at(11) == [-0.9]

    player.reset(seed=bank.get(rep.seed_context["episode"], rep.seed_context["start"]))
    again = player.generate(24, rep.clicks, noise_seed=rep.noise_seed)
    for a, b in zip(first, again):
        assert torch.equal(a, b)


def test_replay_drives_a_different_model(tmp_path):
    """The same recording runs on another model (it is a comparison tool, not a lock)."""
    from play import Player, SessionRecorder, Replay
    bank = _seed_bank()
    seed = bank.pick(np.random.default_rng(6), "val")
    clicks = [(2, 0.8), (9, -0.8)]
    ckpt5b, _ = registry.checkpoint_for(registry.get("flow96c"))
    rec = SessionRecorder("flow96c", ckpt5b, seed, 0)
    for f, x in clicks:
        rec.click(f, x)
    rec.advance(8)
    rep = Replay(rec.save(str(tmp_path / "s.json")))

    ckpt5a, _ = registry.checkpoint_for(registry.get("flow96"))
    player = Player(ckpt5a, bank, device=_device(), name="flow96")
    player.reset(seed=bank.get(rep.seed_context["episode"], rep.seed_context["start"]))
    frames = player.generate(8, rep.clicks, noise_seed=rep.noise_seed)
    assert len(frames) == 8
    assert frames[-1].shape == (3, bank.height, bank.width)


# --------------------------------------------------------------------------- #
# The window (SDL dummy driver: never opens a real window)
# --------------------------------------------------------------------------- #

class _FakePlayer:
    """Just enough of Player for the drawing path, so the window test loads no model."""

    def __init__(self, bank):
        self.clicks = ClickState()
        self.frame_index = 0
        # a real seed from the bank, so switching models (which re-seeds) works
        self.seed = bank.pick(np.random.default_rng(0), "val")
        self.name = "fake"
        self.checkpoint = "fake.pt"
        self.hist_extra, self.K = 8, 4
        self.rng = np.random.default_rng(0)
        self._frame = torch.zeros(3, bank.height, bank.width)

    def reset(self, seed=None, episode=None, start=None):
        self.seed = seed if seed is not None else self.seed
        self.clicks = ClickState()
        self.frame_index = 0
        return self.current_frame()

    def current_frame(self):
        return self._frame

    def click(self, x, frame=None):
        return self.clicks.click(x, self.frame_index if frame is None else frame)

    def step(self, action, seed=None):
        self.frame_index += 1
        return self._frame


def _client(bank, argv_extra=(), model="flow96c"):
    args = play.build_parser().parse_args(
        ["--model", model, "--no-power", *argv_extra])
    catalog = play.Catalog()
    client = play.Client(args, catalog, catalog.cycle(), model, bank)
    client.players = {model: _FakePlayer(bank)}
    return client


@pytest.fixture()
def dummy_display(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.display.quit()
    yield pygame
    pygame.quit()


def test_catalog_resolves_names_aliases_and_bare_runs(tmp_path, monkeypatch):
    cat = play.Catalog()
    assert cat.default() == "flow96c"
    assert cat.cycle() == ["flow96", "flow96c"]
    assert cat.info("5b")["name"] == "flow96c"           # alias
    assert cat.info("flow96d")["name"] == "flow96d"      # registered, not selected
    assert cat.info("flow96")["kind"] == "copy"          # from the frozen copy

    # an unregistered run directory is playable by name
    monkeypatch.setattr(registry, "RUNS_DIR", str(tmp_path))
    (tmp_path / "flow96e").mkdir()
    (tmp_path / "flow96e" / "best.pt").write_bytes(b"x")
    info = cat.info("flow96e")
    assert info["kind"] == "run" and info["name"] == "flow96e"
    assert info["path"].endswith(os.path.join("flow96e", "best.pt"))
    with pytest.raises(registry.RegistryError) as e:
        cat.info("flow96z")
    assert "flow96c" in str(e.value) and "does not exist" in str(e.value)


def test_m_cycles_only_selected_models(dummy_display, monkeypatch):
    """M walks the selected models; an unselected start joins at the first selected one."""
    bank = _seed_bank()
    client = _client(bank, model="flow96c")
    client._open_window()
    client._player_for("flow96c")
    client._reset_run()
    assert client.cycle == ["flow96", "flow96c"]

    client._switch_model(1)
    assert client.model_name == "flow96"                 # wraps within the cycle
    client._switch_model(-1)
    assert client.model_name == "flow96c"

    # starting on the rejected model: M jumps into the selected cycle, and never lands
    # back on flow96d
    other = _client(bank, model="flow96d")
    other._open_window()
    other._player_for("flow96d")
    other._reset_run()
    assert other.model_name == "flow96d"
    other._switch_model(1)
    assert other.model_name == "flow96"
    other._switch_model(1)
    assert other.model_name == "flow96c"
    other._switch_model(1)
    assert other.model_name == "flow96"


def test_switch_keeps_the_seed_context(dummy_display):
    bank = _seed_bank()
    client = _client(bank, model="flow96c")
    client._open_window()
    client._player_for("flow96c")
    client._reset_run(seed=bank.pick(np.random.default_rng(9), "val"))
    before = client.player.seed.label
    client._switch_model(1)
    assert client.player.seed.label == before           # same seed context on both models


def test_window_opens_resizes_and_toggles_fullscreen(dummy_display, tmp_path):
    pygame = dummy_display
    bank = _seed_bank()
    client = _client(bank, ["--scale", "4", "--no-power"])
    assert client.windowed_size == (bank.width * 4, bank.height * 4)
    client._open_window()
    assert pygame.display.get_surface() is not None
    import time
    client._draw(time.perf_counter())
    assert client._rect[2:] == (bank.width * 4, bank.height * 4)     # clean integer scale

    # resize (what an X11/Wayland resize does) -> the rect follows, centred
    client.screen = pygame.display.set_mode((1000, 500), pygame.RESIZABLE)
    client._draw(time.perf_counter())
    assert client._rect[:2] == (212, 34)
    assert client._rect[2:] == (576, 432)

    # fullscreen toggle and back
    client._toggle_fullscreen()
    assert client.fullscreen is True
    client._draw(time.perf_counter())
    left, top, w, h = client._rect
    assert w <= client.screen.get_width() and h <= client.screen.get_height()
    assert left >= 0 and top >= 0
    client._toggle_fullscreen()
    assert client.fullscreen is False
    assert client.screen.get_size() == (1000, 500)


def test_pulses_are_drawn_for_every_click_and_never_fed_to_the_model(dummy_display):
    pygame = dummy_display
    bank = _seed_bank()
    client = _client(bank)
    client._open_window()
    client._rect = (0, 0, bank.width * 4, bank.height * 4)

    def native():
        client._native = None
        return client._compose_native(time_now())

    plain = native().astype(np.int16)
    # two clicks 1 frame apart: the second does not move the signal, both pulse
    client._on_click(client._rect[0])
    client._on_click(client._rect[0] + client._rect[2])
    assert len(client.pulses) == 2
    assert client.player.clicks.x == pytest.approx(-1.0)     # the ignored click changed nothing
    painted = native().astype(np.int16)
    # the two discs are at the left and right image edges, vertically centred
    row = bank.height // 2
    assert painted[row, 0:4].max() > plain[row, 0:4].max()
    assert painted[row, -4:].max() > plain[row, -4:].max()

    client._draw(time_now())
    assert len(client.pulses) == 2
    client.pulses[0][1] -= play.PULSE_SECONDS + 0.01         # age them out
    client.pulses[1][1] -= play.PULSE_SECONDS + 0.01
    client._prune_pulses(time_now())
    assert client.pulses == []
    assert np.array_equal(native(), plain.astype(np.uint8))  # nothing left on the image


def time_now():
    import time
    return time.perf_counter()


def test_hud_reports_the_state_the_brief_asks_for(dummy_display):
    bank = _seed_bank()
    client = _client(bank)
    client._open_window()
    client.show_help = True
    text = " ".join(t for t, _ in client._hud_lines())
    for needle in ("flow96c", "fps", "frame", "held X", "seed", "noise seed", "96x72",
                   "windowed", "nearest"):
        assert needle in text, needle


def test_hud_shows_the_internal_name_and_flags_a_rejected_model(dummy_display):
    bank = _seed_bank()
    client = _client(bank, model="flow96d")
    client._open_window()
    assert client._model_tag() == "flow96d [not selected]"
    # an alias is shown next to the internal name
    assert _client(bank, model="5b")._model_tag() == "flow96c (5b)"


# --------------------------------------------------------------------------- #
# No-oracle source check
# --------------------------------------------------------------------------- #

def test_player_sources_have_no_evaluation_or_engine_dependency():
    """The player must not reach for the judges, the sidecar, or the engine at all."""
    import re
    for name in ("play.py", "model_registry.py", "build_seeds.py", "register_model.py"):
        src = open(os.path.join(HERE, "..", name), encoding="utf-8").read()
        for pat in (r"^\s*import\s+judges", r"^\s*from\s+judges\s+import",
                    r"^\s*import\s+scorecard", r"^\s*from\s+scorecard\s+import",
                    r"^\s*import\s+analysis", r"^\s*from\s+analysis\s+import"):
            assert not re.search(pat, src, re.M), f"{name} imports {pat}"
        assert "sidecar" not in src, f"{name} mentions the sidecar"
        assert "analysis.jsonl" not in src, f"{name} reads the analysis sidecar"
