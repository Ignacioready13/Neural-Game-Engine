"""Stage 6 — the player: a real-time thin client for experiencing and comparing models.

It reads the mouse, applies the engine's hold/debounce semantics, runs one model in a
loop at the training rate (20 Hz), and draws the result. The engine is never involved:
the only things fed to the model are the K most recent frames (a real seed context,
then its own output) and the click-derived signals. The click pulse and every other
overlay are *display only* and are never fed back (N1/N2).

Starting it — one command, no flags to remember::

    ./play.sh                      # newest registered model, windowed
    ./play.sh --model 5a           # pick a model by name
    ./play.sh --list-models        # what is registered
    .venv/bin/python play.py --help

Click semantics mirror the engine (``engine/src/control/steering.ts``):
``actionX = clamp((clientX - imageLeft) / imageWidth * 2 - 1, -1, 1)`` (X only,
relative to the *displayed image*, so clicks in the letterbox clamp to the edges);
a click within 3 frames (150 ms at 20 Hz) of the last **accepted** click does not move
the held signal but still shows its pulse; ``actionHasClicked`` latches to 1 after the
first accepted click. Derived features (deltaActionX, frames-since-click) are computed
by ``steer.click_features``, the same code training used.

The click pulse reproduces the engine's shader (``engine/src/render/Renderer.ts``): a
soft white radial disc at the click's X, vertically centred, whose radius grows from
0.055 to 0.1375 of the image width while ``alpha = 0.85 * (1 - age / 0.4 s)`` fades it
out over 400 ms. It is drawn for every click, accepted or not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch

import model_registry as registry
from dataset import PackedDataset, frames_to_tensor
from eval_harness import features_from_history, model_predict

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW_DIR = os.path.join(HERE, "review", "stage6")

# --- constants read off the engine (read-only reference) --------------------- #
TRAIN_HZ = 20.0
DT = 1.0 / TRAIN_HZ
DEBOUNCE_FRAMES = 3            # engine DEBOUNCE_SECONDS = 0.150 at 20 Hz
PULSE_SECONDS = 0.400          # engine CLICK_PULSE_SECONDS
PULSE_ALPHA = 0.85             # engine: alpha = life * 0.85
PULSE_RX0 = 0.055              # engine: rx = 0.055 * (1 + (1 - life) * 1.5) clip units
PULSE_GROWTH = 1.5
PULSE_STOPS = ((0.0, 1.0), (0.4, 0.7), (1.0, 0.0))   # engine makeDiscTexture gradient

# Catch up at most this many frames per loop pass: beyond it we are not keeping up,
# so drop the excess and say so rather than spiralling.
MAX_CATCHUP = 2


# --------------------------------------------------------------------------- #
# Click state (the engine's semantics)
# --------------------------------------------------------------------------- #

class ClickState:
    """The engine's held-click semantics (debounce 3 frames = 150 ms at 20 Hz)."""

    def __init__(self, debounce_frames: int = DEBOUNCE_FRAMES):
        self.debounce = int(debounce_frames)
        self.x = 0.0
        self.has_clicked = 0.0
        self.last_accept = None
        self.accepted = 0

    def click(self, x: float, now_frame: int) -> bool:
        """Offer a click at frame ``now_frame``. Returns True if it was accepted."""
        x = float(np.clip(x, -1.0, 1.0))
        if self.last_accept is None or (now_frame - self.last_accept) >= self.debounce:
            self.last_accept = int(now_frame)
            self.x = x
            self.has_clicked = 1.0
            self.accepted += 1
            return True
        return False

    def frames_since_accept(self, now_frame: int):
        """Frames since the last accepted click, or None if none has been accepted."""
        return None if self.last_accept is None else int(now_frame) - int(self.last_accept)

    @property
    def action(self) -> tuple[float, float]:
        return (self.x, self.has_clicked)

    def restore(self, hist: np.ndarray) -> None:
        """Reconstruct the state from a real click-signal history.

        The seed context carries the real clicks that preceded it (the model sees
        them), so the debounce state has to be reconstructed too — otherwise the
        first click of a session would be accepted even when it lands one frame after
        a real one.
        """
        hist = np.asarray(hist, dtype=np.float32)
        ax = hist[:, 0]
        n = len(ax)
        last = None
        for i in range(1, n):
            if ax[i] != ax[i - 1]:
                last = i
        self.x = float(ax[-1])
        self.has_clicked = 1.0 if float(hist[-1][1]) > 0.5 else 0.0
        if last is not None:
            # ``last`` is the first row holding the new value; relative to the
            # rollout's frame 0 (the last context frame) that is a negative offset.
            self.last_accept = int(last - (n - 1))
            self.accepted = 1
        elif self.x != 0.0 or self.has_clicked > 0.5:
            # a click was accepted before the retained history (the engine's default
            # mode holds 0), so it is well outside the debounce window
            self.last_accept = -int(n - 1)
            self.accepted = 1


# --------------------------------------------------------------------------- #
# Geometry: letterbox, click mapping, pulse size
# --------------------------------------------------------------------------- #

def image_rect(win_w: int, win_h: int, img_w: int, img_h: int, integer: bool = True):
    """Where the image lands in the window: aspect preserved, centred, letterboxed.

    Returns ``(left, top, width, height, scale)``. With ``integer`` the scale is the
    largest whole number that fits (so nearest-neighbour scaling keeps square pixels);
    when the window is smaller than the image it falls back to a fractional downscale.
    """
    ww, wh, iw, ih = int(win_w), int(win_h), int(img_w), int(img_h)
    if iw <= 0 or ih <= 0 or ww <= 0 or wh <= 0:
        return 0, 0, iw, ih, 1.0
    s_int = min(ww // iw, wh // ih)
    s = float(s_int) if (integer and s_int >= 1) else min(ww / iw, wh / ih)
    w = max(1, int(round(iw * s)))
    h = max(1, int(round(ih * s)))
    return (ww - w) // 2, (wh - h) // 2, w, h, s


def click_to_signal(client_x: float, image_left: float, image_width: float) -> float:
    """The engine's canvas mapping: image-relative X only, clipped to [-1, 1]."""
    if image_width <= 0:
        return 0.0
    t = (float(client_x) - float(image_left)) / float(image_width)
    return float(np.clip(t * 2.0 - 1.0, -1.0, 1.0))


def signal_to_image_x(x: float, image_left: float, image_width: float) -> float:
    """Inverse of :func:`click_to_signal` (unclamped), for placing the pulse."""
    return float(image_left) + (float(x) * 0.5 + 0.5) * float(image_width)


def pulse_radius_px(image_width: float, life: float) -> float:
    """The engine's disc radius in pixels at ``life`` (1 = fresh, 0 = gone)."""
    life = min(1.0, max(0.0, float(life)))
    rx = PULSE_RX0 * (1.0 + (1.0 - life) * PULSE_GROWTH)
    return rx * float(image_width) / 2.0


# --------------------------------------------------------------------------- #
# Frame I/O (used by the display and by headless review output)
# --------------------------------------------------------------------------- #

def _to_uint8(frame) -> np.ndarray:
    """A frame as (H,W,3) uint8, from a torch [-1,1] tensor or a uint8 array.

    Accepting both keeps the writers usable from the model path and from the
    review-image helpers, which keep frames as uint8 (mixing the two silently made
    every "[0,1] units" number in the Stage 5 metrics meaningless, so the dtype is
    decided explicitly here as well).
    """
    if isinstance(frame, np.ndarray):
        if frame.dtype == np.uint8:
            return frame
        return np.clip(frame * 255 if frame.max() <= 1.0 else frame, 0, 255).astype(np.uint8)
    t = frame.float().detach().cpu()
    if t.ndim == 3:
        t = t.permute(1, 2, 0)
    return ((t * 0.5 + 0.5) * 255).clamp(0, 255).numpy().astype(np.uint8)


def write_gif(frames, path: str, scale: int = 6, fps: int = 20, max_frames: int = 400):
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imgs = []
    for f in frames[:max_frames]:
        u = _to_uint8(f)
        im = Image.fromarray(u).resize((u.shape[1] * scale, u.shape[0] * scale),
                                       Image.Resampling.NEAREST)
        imgs.append(im)
    if not imgs:
        return path
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0, optimize=False)
    return path


def write_strip(frames, path: str, every: int = 10, scale: int = 4):
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sel = frames[::every]
    imgs = [Image.fromarray(_to_uint8(f)).resize(
        (_to_uint8(f).shape[1] * scale, _to_uint8(f).shape[0] * scale),
        Image.Resampling.NEAREST) for f in sel]
    if not imgs:
        return path
    w, h = imgs[0].size
    cols = min(6, len(imgs))
    rows = int(np.ceil(len(imgs) / cols))
    sheet = Image.new("RGB", (cols * (w + 4), rows * (h + 4)), (20, 20, 20))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * (w + 4), (i // cols) * (h + 4)))
    sheet.save(path)
    return path


# --------------------------------------------------------------------------- #
# Seed contexts
# --------------------------------------------------------------------------- #

class Seed(dict):
    """A seed context: K real frames plus the real click history that precedes them."""

    @property
    def label(self) -> str:
        return f"{self['split']} ep{self['episode']}@{self['start']}"


class SeedBank:
    """Where a run's opening K real frames come from (packed set, or the seed cache)."""

    width = 0
    height = 0
    kind = "none"
    describe = "no seeds"

    def pick(self, rng, split: str | None = None) -> Seed:      # pragma: no cover - interface
        raise NotImplementedError

    def get(self, episode: int, start: int) -> Seed | None:      # pragma: no cover - interface
        raise NotImplementedError


class PackedSeeds(SeedBank):
    """Seeds read straight out of a packed dataset (needs the packed set present)."""

    kind = "packed"

    def __init__(self, packed_path: str, split: str = "val", hist_extra: int = 8, K: int = 4):
        self.packed = PackedDataset(packed_path)
        self.split = split
        self.hist_extra = int(hist_extra)
        self.K = int(K)
        self.width, self.height = self.packed.width, self.packed.height
        self.describe = f"packed {os.path.basename(os.path.normpath(packed_path))}"

    def _episodes(self, split: str):
        eps = [e for e in self.packed.episodes_by_split[split] if e["length"] > self.K + 20]
        if not eps:
            raise SystemExit(f"split {split!r} has no usable episodes in {self.packed.path}")
        return eps

    def _make(self, e: dict, start: int, split: str) -> Seed:
        off = int(e["frame_offset"])
        start = int(start)
        lo = max(0, start - self.hist_extra)
        h = self.packed.action_vectors(self.packed.labels[off + lo: off + start + self.K])
        h = np.nan_to_num(h, nan=0.0).astype(np.float32)
        if h.shape[0] < self.hist_extra + self.K:            # episode starts too early
            h = np.concatenate(
                [np.zeros((self.hist_extra + self.K - h.shape[0], 2), np.float32), h])
        return Seed(episode=int(e["episode_index"]), start=start, split=split,
                    frames=np.asarray(self.packed.images[off + start: off + start + self.K]),
                    hist=h)

    def pick(self, rng, split: str | None = None) -> Seed:
        split = split or self.split
        eps = self._episodes(split)
        e = eps[int(rng.integers(len(eps)))]
        L = int(e["length"])
        start = int(rng.integers(0, max(1, int(0.1 * L))))
        return self._make(e, start, split)

    def get(self, episode: int, start: int) -> Seed | None:
        if episode is None or episode >= len(self.packed.episodes):
            return None
        e = self.packed.episodes[int(episode)]
        if int(start) < 0 or int(start) + self.K > int(e["length"]):
            return None
        return self._make(e, int(start), e["split"])


class CacheSeeds(SeedBank):
    """Seeds from ``models/seeds.npz`` (built by build_seeds.py) — no packed set needed."""

    kind = "cache"

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise SystemExit(
                f"no seed cache at {path}; build one with "
                "`.venv/bin/python build_seeds.py`, or pass --packed <packed set>")
        z = np.load(path, allow_pickle=False)
        self.frames = z["frames"]
        self.hist = z["hist"]
        self.episode = z["episode"]
        self.start = z["start"]
        self.split = z["split"]
        self.K = int(z["K"])
        self.hist_extra = int(z["hist_extra"])
        self.width, self.height = int(z["width"]), int(z["height"])
        self.packed = str(z["packed"])
        self.describe = f"cache {os.path.basename(path)} ({len(self.episode)} seeds)"

    def _make(self, i: int) -> Seed:
        return Seed(episode=int(self.episode[i]), start=int(self.start[i]),
                    split=str(self.split[i]), frames=np.asarray(self.frames[i]),
                    hist=np.asarray(self.hist[i]))

    def splits(self) -> list[str]:
        return sorted(set(str(s) for s in self.split))

    def pick(self, rng, split: str | None = None) -> Seed:
        idx = np.arange(len(self.episode))
        if split:
            keep = idx[self.split == split]
            if len(keep) == 0:
                raise SystemExit(f"the seed cache has no {split!r} seeds; "
                                 f"pass --packed to use a split directly")
            idx = keep
        return self._make(int(idx[int(rng.integers(len(idx)))]))

    def get(self, episode: int, start: int) -> Seed | None:
        if episode is None:
            return None
        hit = np.where((self.episode == int(episode)) & (self.start == int(start)))[0]
        return self._make(int(hit[0])) if len(hit) else None


def default_seed_packed(reg: dict | None = None) -> str:
    """The packed set the registry's models were seeded from (first one recorded)."""
    reg = reg if reg is not None else registry.load()
    for m in registry.entries(reg, selected_only=False):
        if m.get("seed_packed"):
            return m["seed_packed"]
    return "data/bulk_v1_96x72"


def open_seed_bank(args, reg: dict | None = None) -> SeedBank:
    """``--packed`` wins when given; otherwise the cache, which needs no packed set."""
    if args.packed:
        return PackedSeeds(args.packed, args.split, hist_extra=args.hist_extra)
    try:
        bank = CacheSeeds(args.seeds)
    except SystemExit:
        fallback = registry.resolve(default_seed_packed(reg))
        print(f"[play] no seed cache; falling back to the packed set {fallback}",
              file=sys.stderr)
        return PackedSeeds(fallback, args.split, hist_extra=args.hist_extra)
    if args.split not in bank.splits():
        print(f"[play] note: the seed cache has no {args.split!r} seeds; "
              f"using {bank.splits()}", file=sys.stderr)
        args.split = bank.splits()[0]
    return bank


# --------------------------------------------------------------------------- #
# Resolving a model name to a checkpoint
# --------------------------------------------------------------------------- #

class Catalog:
    """Resolves a ``--model`` name to a checkpoint path.

    Accepts, in order: a registered internal name (``flow96c``), a registered alias
    (``5b``), or a bare run name whose ``runs/<name>/best.pt`` exists but which was never
    registered — so every flow96-family model can be played, selected or not, without
    registering it first.
    """

    def __init__(self, reg: dict | None = None):
        self.reg = reg if reg is not None else registry.load()
        self._cache: dict[str, dict] = {}

    def playable(self, name: str) -> bool:
        """True when ``name`` resolves to a checkpoint that is present in this checkout.

        The registry is provenance and lists every revision, but a checkout only ships
        the checkpoints it actually carries (this repository ships ``flow96e``), so a
        registered name can be unplayable here.
        """
        try:
            return os.path.exists(self.info(name)["path"])
        except registry.RegistryError:
            return False

    def cycle(self) -> list[str]:
        """The names the ``M`` key cycles: selected models present in this checkout."""
        return [n for n in registry.names(self.reg, selected_only=True)
                if self.playable(n)]

    def default(self) -> str:
        """The newest *playable* selected model, else the newest playable one."""
        cycle = self.cycle()
        if cycle:
            return cycle[-1]
        for name in reversed(registry.names(self.reg)):
            if self.playable(name):
                return name
        return registry.newest(self.reg)["name"]

    def info(self, name: str) -> dict:
        """``{"name", "path", "kind", "entry", "aliases"}`` for a resolvable name."""
        if name in self._cache:
            return self._cache[name]
        entry = registry.find(name, self.reg)
        if entry is not None:
            path, kind = registry.checkpoint_for(entry)
            info = {"name": entry["name"], "path": path, "kind": kind, "entry": entry,
                    "aliases": list(entry.get("aliases") or [])}
        else:
            path = registry.run_checkpoint(name)
            if not os.path.exists(path):
                known = ", ".join(registry.names(self.reg)) or "(none registered)"
                raise registry.RegistryError(
                    f"unknown model {name!r}. Registered models: {known}. "
                    f"(A bare run name works too, e.g. `--model flow96e` for "
                    f"runs/flow96e/best.pt, but {registry.rel(path)} does not exist.)")
            info = {"name": name, "path": path, "kind": "run", "entry": None,
                    "aliases": []}
        self._cache[name] = info
        return info

    def known(self) -> list[str]:
        return registry.names(self.reg, selected_only=False)

    def table(self) -> str:
        return registry.format_table(self.reg)



# --------------------------------------------------------------------------- #
# The player (model + closed loop)
# --------------------------------------------------------------------------- #

class Player:
    """Generates frames in closed loop from a seed context + click signals."""

    def __init__(self, checkpoint: str, seeds: SeedBank, device: str | None = None,
                 hist_extra: int = 8, name: str | None = None, split: str | None = None,
                 noise_corr: float = 0.0):
        from gates import load_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, payload = load_model(checkpoint, self.device)
        mc = payload.get("model_config", {})
        self.K = int(mc.get("K", seeds.K))
        self.hist_extra = int(max(1, hist_extra))
        self.seeds = seeds
        self.prefer_split = split
        self.name = name or os.path.basename(os.path.dirname(checkpoint)) or checkpoint
        self.checkpoint = checkpoint
        self.rng = np.random.default_rng()
        self.clicks = ClickState()
        self.frame_index = 0
        self.session_frames = 0
        self.seed: Seed | None = None
        # Stage 5c, additive: 0.0 (the default) is exactly today's fresh independent
        # noise every frame. > 0 makes the flow noise an AR(1) process,
        # noise_f = rho*noise_{f-1} + sqrt(1-rho^2)*fresh_f, still a deterministic
        # function of (noise_seed, f) so replay stays exact.
        self.noise_corr = float(noise_corr or 0.0)
        self._noise_z = None

    # -- power accounting -------------------------------------------------- #

    def start_session(self):
        """Begin sampling power for this play session (writes a ledger row on exit)."""
        from powerlog import PowerLogger
        self._plog = getattr(self, "_plog", None) or PowerLogger()
        self._plog.start()
        self.session_frames = 0
        return self

    def power_latest(self):
        """The most recent power sample for a live readout (None when not sampling)."""
        plog = getattr(self, "_plog", None)
        return plog.latest() if plog is not None else None

    def end_session(self, run_id="play", notes=None, write=True):
        """Stop sampling; print the one-line summary and append a ledger record."""
        plog = getattr(self, "_plog", None)
        if plog is None:
            return None
        rec = plog.stop()
        n = self.session_frames
        dur = max(1e-6, rec.get("duration_s") or 0.0)
        rec.update(run_id=run_id, purpose="play", notes=notes, n_frames=n,
                   fps=round(n / dur, 2),
                   j_per_frame=(round((rec.get("gpu_energy_j") or 0.0) / n, 4) if n else None),
                   wh_per_hour_of_play=rec.get("mean_gpu_w"),
                   w_per_fps=(round(rec["mean_gpu_w"] / (n / dur), 4)
                              if rec.get("mean_gpu_w") and n else None),
                   parent_run=(notes or ""))
        if write:
            from powerlog import append_record
            append_record(rec)
        print(f"[play] {n} frames in {dur:.1f}s ({n / dur:.1f} fps)  "
              f"GPU {rec.get('mean_gpu_w')} W mean / {rec.get('peak_gpu_w')} W peak  "
              f"{rec.get('j_per_frame')} J/frame  "
              f"{rec.get('wh_per_hour_of_play')} Wh per hour of play  "
              f"GPU {rec.get('mean_gpu_temp_c')} C  throttle={rec.get('throttle_events')}",
              flush=True)
        return rec

    # -- seeding ----------------------------------------------------------- #

    def reset(self, seed: Seed | None = None, episode: int | None = None,
              start: int | None = None):
        """Seed the run with K real frames (a random early-episode clip by default)."""
        self._noise_z = None
        if seed is None:
            if episode is None or start is None:
                seed = self.seeds.pick(self.rng, self.prefer_split)
            else:
                seed = self.seeds.get(episode, start)
                if seed is None:
                    raise SystemExit(
                        f"seed context {episode}:{start} is not available in "
                        f"{self.seeds.describe}; pass --packed to use it directly")
        elif not isinstance(seed, Seed) or "frames" not in seed:
            # a seed may arrive as an identifier only (e.g. from a recording, or from
            # another player): look the frames up in this bank
            refetched = self.seeds.get(seed["episode"], seed["start"])
            if refetched is None:
                raise SystemExit(
                    f"seed context {seed['episode']}:{seed['start']} is not in "
                    f"{self.seeds.describe}; pass --packed to use it directly")
            seed = refetched
        self.seed = seed
        self.episode, self.start = int(seed["episode"]), int(seed["start"])
        self.seed_split = seed.get("split")
        self.frames = frames_to_tensor(np.asarray(seed["frames"])).to(self.device)
        self.hist = [np.asarray(r, dtype=np.float32) for r in seed["hist"]]
        self.clicks = ClickState()
        self.clicks.restore(np.stack(self.hist))
        self.frame_index = 0
        self._trim_history()
        return self.current_frame()

    def _trim_history(self):
        """Keep only what the features need (the last K clicks need 8 before them).

        Without this, ``steer.click_features`` walks the whole growing history in a
        Python loop every frame, so a long session slows down without bound.
        """
        keep = self.hist_extra + self.K
        if len(self.hist) > keep:
            del self.hist[:len(self.hist) - keep]

    # -- generation -------------------------------------------------------- #

    def current_frame(self):
        return self.frames[-1]

    @torch.no_grad()
    def step(self, action: tuple[float, float], seed: int | None = None):
        """Advance one frame. ``action`` = the click-derived (actionX, hasClicked)."""
        self.hist.append(np.array([action[0], action[1]], dtype=np.float32))
        self._trim_history()
        feats = features_from_history(np.stack(self.hist), self.K, self.device)
        ctx = self.frames[-self.K:].unsqueeze(0)
        # the action tensor is the real click history, not the current action repeated
        a = torch.from_numpy(np.ascontiguousarray(
            np.stack(self.hist[-self.K:])))[None].to(self.device)
        if self.noise_corr > 0.0 and hasattr(self.model, "predict"):
            noise = self._correlated_noise(seed)
            out = self.model.predict(ctx, a, features=feats, seed=seed, noise=noise)
        else:
            out = model_predict(self.model, ctx, a, feats, seed, return_shift=True)
        nxt = out[0] if isinstance(out, tuple) else out
        self.frames = torch.cat([self.frames, nxt], dim=0)
        self.frame_index += 1
        self.session_frames += 1
        return nxt[0]

    def _correlated_noise(self, seed: int | None):
        """AR(1) flow noise for this frame; deterministic in (noise_seed, frame)."""
        h, w = self.frames.shape[-2:]
        g = torch.Generator(device="cpu").manual_seed(int(seed or 0))
        fresh = torch.randn((1, 3, h, w), generator=g)
        rho = self.noise_corr
        z = (fresh if self._noise_z is None
             else rho * self._noise_z + (1.0 - rho * rho) ** 0.5 * fresh)
        self._noise_z = z
        return z

    def click(self, x: float, frame: int | None = None) -> bool:
        """Offer a click; returns True if it moved the held signal."""
        return self.clicks.click(x, self.frame_index if frame is None else frame)

    def generate(self, n_frames: int, click_script=None, noise_seed: int = 0):
        """Generate ``n_frames`` new frames with a scripted ``[(frame, x)]`` click list."""
        script: dict[int, list[float]] = {}
        for cf, cx in (click_script or []):
            script.setdefault(int(cf), []).append(float(cx))
        frames = []
        for f in range(n_frames):
            for cx in script.get(f, ()):
                self.click(cx, f)
            frames.append(self.step(self.clicks.action, seed=int(noise_seed) + f))
        return frames


# --------------------------------------------------------------------------- #
# Session recording / replay
# --------------------------------------------------------------------------- #

RECORD_KIND = "nge-player-session"


class SessionRecorder:
    """Records (model, seed context, noise seed, click log) so a run can be replayed."""

    def __init__(self, model: str, checkpoint: str, seed: Seed, noise_seed: int,
                 notes: str = ""):
        self.data = {
            "kind": RECORD_KIND, "version": 1,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model, "checkpoint": registry.rel(checkpoint),
            "seed_context": {"episode": int(seed["episode"]), "start": int(seed["start"]),
                             "split": seed.get("split"), "label": seed.label},
            "noise_seed": int(noise_seed),
            "resolution": [int(seed["frames"].shape[2]), int(seed["frames"].shape[1])],
            "fps": TRAIN_HZ,
            "clicks": [],
            "n_frames": 0,
            "notes": notes,
        }

    def click(self, frame: int, x: float):
        self.data["clicks"].append([int(frame), round(float(x), 6)])

    def advance(self, n_frames: int):
        self.data["n_frames"] = int(n_frames)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, path)
        return path


class Replay:
    """A recorded session's click log, ready to drive any model."""

    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        if self.data.get("kind") != RECORD_KIND:
            raise SystemExit(f"{path} is not a {RECORD_KIND} file")
        self.path = path
        self.clicks = sorted((int(f), float(x)) for f, x in self.data.get("clicks", []))
        self.n_frames = int(self.data.get("n_frames") or 0)
        self.noise_seed = int(self.data.get("noise_seed") or 0)
        self.seed_context = self.data.get("seed_context") or {}
        self.model = self.data.get("model")
        self.notes = self.data.get("notes", "")

    def clicks_at(self, frame: int) -> list[float]:
        return [x for f, x in self.clicks if f == frame]


def default_recording_path(model: str, replay: bool = False) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    kind = "replay" if replay else "session"
    return os.path.join(registry.RECORDINGS_DIR, f"{model}_{kind}_{stamp}.json")


# --------------------------------------------------------------------------- #
# The interactive client
# --------------------------------------------------------------------------- #

class Client:
    """The window: input, 20 Hz pacing, letterboxed display, pulses, HUD, replay."""

    def __init__(self, args, catalog: Catalog, cycle: list[str], start: str,
                 seed_bank: SeedBank):
        self.args = args
        self.catalog = catalog
        self.cycle = list(cycle)          # the selected models M cycles through
        self.current = catalog.info(start)["name"]
        self.seeds = seed_bank
        self.players: dict[str, Player] = {}
        self.pulses: list[list] = []          # [image_x, t0]
        self.recorder: SessionRecorder | None = None
        self.replay: Replay | None = None
        self.display_rec: list[np.ndarray] | None = None
        self._display_rec_path = None
        self._recording_path = None
        self.paused = False
        self.show_hud = not args.no_hud
        self.show_help = bool(args.help_overlay)
        self.integer_scale = not args.smooth
        self.fullscreen = bool(args.fullscreen)
        w, h = args.window_w, args.window_h
        if not w or not h:
            s = int(args.scale or 8)
            w, h = seed_bank.width * s, seed_bank.height * s
        self.windowed_size = (int(w), int(h))
        self.noise_seed = int(args.noise_seed)
        self.seed_override: Seed | None = None
        self._acc = 0.0
        self._last = time.perf_counter()
        self._dirty = True
        self._dropped = 0
        self._gen_ms = 0.0
        self._last_warn = 0.0
        self._gen_times: deque[float] = deque(maxlen=128)
        self._lat: deque[float] = deque(maxlen=1000)
        self.stat_lines: list[list] = []       # [text, seconds]

    # -- lifecycle --------------------------------------------------------- #

    def _player_for(self, name: str) -> Player:
        if name not in self.players:
            info = self.catalog.info(name)
            if info["kind"] == "source" and not self.players:
                print(f"[play] WARNING: {name} has no frozen copy; loading "
                      f"{registry.rel(info['path'])} (a later training run could "
                      f"overwrite what {name} means)", file=sys.stderr)
            elif info["kind"] == "run" and not self.players:
                print(f"[play] note: {name} is not in the registry; loading "
                      f"{registry.rel(info['path'])} directly", file=sys.stderr)
            print(f"[play] loading model {name} ...", flush=True)
            self.players[name] = Player(info["path"], self.seeds, device=self.args.device,
                                        hist_extra=self.args.hist_extra, name=name,
                                        split=self.args.split,
                                        noise_corr=float(getattr(self.args, "noise_corr", 0.0)))
        return self.players[name]

    @property
    def player(self) -> Player:
        return self.players[self.current]

    @property
    def model_name(self) -> str:
        return self.current

    def _reset_run(self, seed: Seed | None = None, noise_seed: int | None = None,
                   keep_replay: bool = False):
        self.player.reset(seed=seed)
        if noise_seed is not None:
            self.noise_seed = int(noise_seed)
        self.pulses.clear()
        self._acc = 0.0
        self._last = time.perf_counter()
        self._dirty = True
        if not keep_replay:
            self.replay = None

    def _switch_model(self, step: int):
        """Cycle to the next/previous *selected* model (a non-selected one is left behind)."""
        if not self.cycle:
            self._set_status("no selected models to cycle (use --model to pick one)")
            return
        if len(self.cycle) == 1 and self.current == self.cycle[0]:
            self._set_status(f"only one selected model ({self.cycle[0]}); "
                             f"play another with --model", 3.0)
            return
        if self.current in self.cycle:
            i = self.cycle.index(self.current)
            nxt = self.cycle[(i + step) % len(self.cycle)]
        else:
            # started on an unselected model: jump to the first (or last) selected one
            nxt = self.cycle[0] if step >= 0 else self.cycle[-1]
        if nxt == self.current:
            return
        old_seed = getattr(self.player, "seed", None)
        self.current = nxt
        self._player_for(self.current)
        # Restart from the same seed context and the same noise seed (and, if a replay
        # is running, the same recorded clicks from frame 0), so the only thing that
        # changed is the model.
        replaying = self.replay is not None
        self._reset_run(seed=old_seed, noise_seed=self.noise_seed, keep_replay=True)
        self._set_status(f"{self.model_name}: " + ("replaying the same clicks"
                                                  if replaying else
                                                  "same seed context, same noise seed"),
                         4.0 if replaying else 3.5)

    def _set_status(self, text: str, seconds: float = 2.5):
        self.stat_lines.append([text, seconds])
        del self.stat_lines[:-3]
        self._dirty = True

    # -- window ------------------------------------------------------------ #

    def _open_window(self):
        import pygame
        pygame.init()
        flags = pygame.RESIZABLE | (pygame.FULLSCREEN if self.fullscreen else 0)
        size = (0, 0) if self.fullscreen else self.windowed_size
        self.screen = pygame.display.set_mode(size, flags)
        pygame.display.set_caption(
            f"NeuralGameEngine player — {self._model_tag()} (click to steer, H for help)")
        self.font = pygame.font.SysFont(None, 18)
        self.font_big = pygame.font.SysFont(None, 24)
        self._disc_cache = {}
        self._native = None
        self._scale = 1.0
        self._rect = (0, 0, self.seeds.width, self.seeds.height)
        self._fsurf = None
        self._fsurf_frame = None

    def _toggle_fullscreen(self):
        import pygame
        if not self.fullscreen:
            self.windowed_size = self.screen.get_size()
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.fullscreen = True
        else:
            self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
            self.fullscreen = False
        self._dirty = True
        self._set_status("fullscreen " + ("on" if self.fullscreen else "off"))

    # -- pulses ------------------------------------------------------------ #

    def _disc(self, size: int, alpha: int):
        """A cached engine-style disc: white, soft radial alpha, scaled to ``size``."""
        import pygame
        size = int(max(2, size))
        alpha = int(max(0, min(255, alpha)))
        key = (size, alpha // 4)
        hit = self._disc_cache.get(key)
        if hit is not None:
            return hit
        base = self._disc_cache.get(("base", size))
        if base is None:
            base = pygame.Surface((size, size), pygame.SRCALPHA)
            a = pygame.surfarray.pixels_alpha(base)
            x = np.arange(size)[:, None]
            y = np.arange(size)[None, :]
            r = np.sqrt((x - (size - 1) / 2.0) ** 2 + (y - (size - 1) / 2.0) ** 2) / (size / 2.0)
            ramp = np.interp(r, [s[0] for s in PULSE_STOPS], [s[1] for s in PULSE_STOPS])
            a[:, :] = np.where(r <= 1.0, ramp * 255.0, 0.0).astype(np.uint8)
            del a
            rgb = pygame.surfarray.pixels3d(base)
            rgb[:, :, :] = 255
            del rgb
            self._disc_cache[("base", size)] = base
        hit = base.copy()
        hit.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)
        if len(self._disc_cache) < 512:
            self._disc_cache[key] = hit
        return hit

    def _add_pulse(self, x: float):
        self.pulses.append([float(np.clip(x, -1, 1)), time.perf_counter()])

    def _prune_pulses(self, now: float) -> bool:
        alive = [p for p in self.pulses if now - p[1] < PULSE_SECONDS]
        changed = len(alive) != len(self.pulses)
        self.pulses = alive
        return changed

    def _draw_pulses(self, surface, rect, now: float, clip: bool = True):
        import pygame
        left, top, w, h = rect
        cy = top + h // 2
        if clip:
            surface.set_clip(pygame.Rect(left, top, w, h))
        for x, t0 in self.pulses:
            life = 1.0 - (now - t0) / PULSE_SECONDS
            if life <= 0:
                continue
            size = int(round(pulse_radius_px(w, life) * 2.0))
            cx = signal_to_image_x(x, left, w)
            surface.blit(self._disc(size, int(round(255 * PULSE_ALPHA * life))),
                         (int(round(cx - size / 2.0)), int(round(cy - size / 2.0))))
        if clip:
            surface.set_clip(None)

    # -- pacing / stepping ------------------------------------------------- #

    def _advance(self) -> bool:
        p = self.player
        f = p.frame_index
        if self.replay is not None:
            if f >= self.replay.n_frames:
                self._set_status(f"replay finished ({self.replay.n_frames} frames)", 4.0)
                self.replay = None
                return False
            for x in self.replay.clicks_at(f):
                p.click(x, f)
                self._add_pulse(x)
        t0 = time.perf_counter()
        p.step(p.clicks.action, seed=self.noise_seed + f)
        ms = (time.perf_counter() - t0) * 1000.0
        self._lat.append(ms)
        self._gen_ms = ms if not self._gen_ms else 0.9 * self._gen_ms + 0.1 * ms
        now = time.perf_counter()
        self._gen_times.append(now)
        if self.recorder is not None:
            self.recorder.advance(p.frame_index)
        if self.display_rec is not None:
            self.display_rec.append(self._compose_native(now))
        self._dirty = True
        return True

    def _current_fps(self) -> float:
        if not self._gen_times:
            return 0.0
        now = time.perf_counter()
        t = [x for x in self._gen_times if now - x < 2.0]
        if len(t) < 2:
            return 0.0
        span = t[-1] - t[0]
        return (len(t) - 1) / span if span > 0 else 0.0

    def _update_pacing(self):
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if dt > 0.5 or dt < 0:            # woke from a stall (drag, debugger, sleep)
            dt = 0.0
            self._acc = 0.0
        if self.paused:
            self._acc = 0.0
            return
        self._acc += dt
        steps = int(self._acc // DT)
        if steps <= 0:
            return
        self._acc -= steps * DT
        if steps > MAX_CATCHUP:
            over = steps - MAX_CATCHUP
            self._dropped += over
            steps = MAX_CATCHUP
            self._acc = 0.0
            if now - self._last_warn > 2.0:
                self._last_warn = now
                self._set_status(f"cannot keep up ({self._current_fps():.0f} fps): "
                                 f"dropped {over} step(s)", 3.0)
        for _ in range(steps):
            if not self._advance():
                break

    # -- input ------------------------------------------------------------- #

    def _handle_event(self, ev) -> bool:
        import pygame
        if ev.type == pygame.QUIT:
            return False
        if ev.type in (getattr(pygame, "VIDEORESIZE", -1),
                       getattr(pygame, "WINDOWRESIZED", -2),
                       getattr(pygame, "WINDOWSIZECHANGED", -3)):
            # the letterbox rect is recomputed every draw; force one so a drag shows
            # the new size immediately instead of at the next generated frame
            self._dirty = True
            return True
        if ev.type == pygame.KEYDOWN:
            return self._handle_key(ev)
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self._on_click(ev.pos[0])
        return True

    def _compute_rect(self):
        """The image rect for the window as it is *now* (cheap; used for click mapping)."""
        win = self.screen.get_size()
        return image_rect(win[0], win[1], self.seeds.width, self.seeds.height,
                          self.integer_scale)[:4]

    def _on_click(self, client_x: float):
        p = self.player
        self._rect = self._compute_rect()
        x = click_to_signal(client_x, self._rect[0], self._rect[2])
        if self.replay is not None:
            # during a replay the recorded clicks drive the run; a live click would
            # break the comparison, so it only draws its pulse
            self._add_pulse(x)
            self._set_status("live input ignored during replay (R to take control)", 3.0)
            return
        frame = p.frame_index
        p.click(x, frame)
        self._add_pulse(x)                 # every click shows its pulse
        if self.recorder is not None:
            self.recorder.click(frame, x)

    def _handle_key(self, ev) -> bool:
        import pygame
        k, mod = ev.key, ev.mod
        if k == pygame.K_ESCAPE:
            if self.fullscreen:
                self._toggle_fullscreen()
            else:
                return False
        elif k == pygame.K_F11 or (k == pygame.K_RETURN and mod & pygame.KMOD_ALT):
            self._toggle_fullscreen()
        elif k == pygame.K_q:
            return False
        elif k == pygame.K_m:
            self._switch_model(-1 if mod & pygame.KMOD_SHIFT else 1)
        elif k == pygame.K_r:
            if mod & pygame.KMOD_SHIFT:      # identical rerun: same context, same noise
                self._reset_run(seed=self.player.seed, noise_seed=self.noise_seed)
                self._set_status("restarted: same seed context, same noise seed", 3.5)
            else:                            # fresh: new context and new noise
                self._reset_run(seed=None,
                                noise_seed=int(self.player.rng.integers(1 << 30)))
                self._set_status("new random seed context", 3.5)
        elif k == pygame.K_SPACE:
            self.paused = not self.paused
            self._set_status("paused" if self.paused else "running")
        elif k in (pygame.K_PERIOD, pygame.K_RIGHT):
            self.paused = True
            self._advance()
            self._set_status(f"single step (frame {self.player.frame_index})")
        elif k == pygame.K_TAB:
            self.show_hud = not self.show_hud
            self._dirty = True
        elif k in (pygame.K_h, pygame.K_F1):
            self.show_help = not self.show_help
            self._dirty = True
        elif k == pygame.K_t:
            self.integer_scale = not self.integer_scale
            self._set_status("scaling: " + ("nearest (integer)" if self.integer_scale
                                            else "smooth"))
        elif k in (pygame.K_s, pygame.K_F2):
            self._screenshot()
        elif k == pygame.K_c:
            self._toggle_recording()
        elif k == pygame.K_v:
            self._toggle_display_recording()
        elif k == pygame.K_p:
            print(self._status_text(), flush=True)
        return True

    # -- recording --------------------------------------------------------- #

    def _toggle_recording(self):
        if self.recorder is not None:
            path = self.recorder.save(self._recording_path)
            print(f"[play] recording saved: {path}", flush=True)
            self._set_status(f"recording saved: {os.path.relpath(path, HERE)}", 4.0)
            self.recorder = None
        else:
            self._recording_path = default_recording_path(self.model_name)
            self.recorder = SessionRecorder(self.model_name, self.player.checkpoint,
                                            self.player.seed, self.noise_seed)
            print(f"[play] recording -> {self._recording_path}", flush=True)
            self._set_status("recording every click — C again to save", 4.0)

    def _toggle_display_recording(self):
        if self.display_rec is not None:
            frames, path = self.display_rec, self._display_rec_path
            self.display_rec = None
            write_gif(frames, path, scale=self._gif_scale(), fps=int(TRAIN_HZ))
            print(f"[play] display recording saved: {path} ({len(frames)} frames)", flush=True)
            self._set_status(f"display recording: {os.path.relpath(path, HERE)}", 4.0)
        else:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self._display_rec_path = os.path.join(
                REVIEW_DIR, f"display_{self.model_name}_{stamp}.gif")
            self.display_rec = []
            self._set_status("recording the display — V again to save", 4.0)

    def _screenshot(self):
        import pygame
        os.makedirs(REVIEW_DIR, exist_ok=True)
        path = os.path.join(REVIEW_DIR,
                            f"shot_{self.model_name}_{time.strftime('%Y%m%d-%H%M%S')}.png")
        pygame.image.save(self.screen, path)
        print(f"[play] screenshot: {path}", flush=True)
        self._set_status(f"screenshot: {os.path.relpath(path, HERE)}", 4.0)

    def _gif_scale(self) -> int:
        """How much to upscale the native-resolution recording (keep the file sane)."""
        return max(2, min(6, int(self._scale) or 2))

    def _compose_native(self, now: float) -> np.ndarray:
        """The current frame at native resolution with the pulses on it (for GIFs)."""
        import pygame
        W, H = self.seeds.width, self.seeds.height
        if self._native is None:
            self._native = pygame.Surface((W, H))
        self._native.blit(self._frame_surface(), (0, 0))
        self._draw_pulses(self._native, (0, 0, W, H), now, clip=False)
        return np.transpose(pygame.surfarray.array3d(self._native), (1, 0, 2))

    # -- drawing ----------------------------------------------------------- #

    def _frame_surface(self):
        import pygame
        if self._fsurf_frame != self.player.frame_index or self._fsurf is None:
            u = _to_uint8(self.player.current_frame())
            self._fsurf = pygame.surfarray.make_surface(np.transpose(u, (1, 0, 2)))
            self._fsurf_frame = self.player.frame_index
        return self._fsurf

    def _draw(self, now: float):
        import pygame
        win = self.screen.get_size()
        left, top, w, h, scale = image_rect(win[0], win[1], self.seeds.width,
                                            self.seeds.height, self.integer_scale)
        self._rect = (left, top, w, h)
        self._scale = scale
        src = self._frame_surface()
        if self.integer_scale and scale >= 1:
            scaled = pygame.transform.scale(src, (w, h))
        else:
            scaled = pygame.transform.smoothscale(src, (w, h))
        self.screen.fill((0, 0, 0))
        self.screen.blit(scaled, (left, top))
        self._draw_pulses(self.screen, self._rect, now)
        self._draw_hud()
        pygame.display.flip()

    def _model_tag(self) -> str:
        """The model label: the internal name, plus an alias or a 'not selected' note."""
        info = self.catalog.info(self.model_name)
        bits = [self.model_name]
        if info["aliases"]:
            bits.append(f"({'/'.join(info['aliases'])})")
        if info["entry"] is None:
            bits.append("[unregistered run]")
        elif not info["entry"].get("selected"):
            bits.append("[not selected]")
        return " ".join(bits)

    def _status_text(self) -> str:
        p = self.player
        sec = p.clicks.frames_since_accept(p.frame_index)
        return (f"[play] model={self.model_name} frames={p.frame_index} "
                f"fps={self._current_fps():.1f} frame_ms={self._gen_ms:.1f} "
                f"heldX={p.clicks.x:+.2f} since_click={sec} "
                f"seed={p.seed.label if p.seed else '?'} noise={self.noise_seed} "
                f"res={self.seeds.width}x{self.seeds.height} fs={self.fullscreen} "
                f"dropped={self._dropped}")

    def _power_line(self):
        try:
            latest = self.player.power_latest()
        except Exception:
            return None
        if not latest or latest.get("gpu_w") is None:
            return None
        temp = latest.get("gpu_temp_c")
        return f"GPU {latest['gpu_w']:.0f} W" + (f"  {temp:.0f} C" if temp is not None else "")

    def _hud_lines(self) -> list[tuple[str, tuple[int, int, int]]]:
        p = self.player
        fps = self._current_fps()
        warn = bool(fps and fps < TRAIN_HZ - 1.0) or self._dropped > 0
        sec = p.clicks.frames_since_accept(p.frame_index)
        white, dim, red, green = (235, 235, 235), (170, 170, 170), (255, 90, 90), (140, 255, 140)
        if self.show_help:
            lines = [
                (f"NeuralGameEngine player — {self._model_tag()}", white),
                (f"seed {p.seed.label if p.seed else '?'}   noise seed {self.noise_seed}   "
                 f"frame {p.frame_index}   {self.seeds.width}x{self.seeds.height}   "
                 f"{'PAUSED' if self.paused else 'running'}   "
                 f"{'fullscreen' if self.fullscreen else 'windowed'}   "
                 f"{'nearest' if self.integer_scale else 'smooth'}", dim),
                (f"{fps:4.1f} fps (target {TRAIN_HZ:.0f})   frame {self._gen_ms:4.1f} ms"
                 + (f"   DROPPED {self._dropped} steps" if self._dropped else ""),
                 red if warn else dim),
                (f"held X {p.clicks.x:+.2f}   hasClicked {int(p.clicks.has_clicked)}   "
                 f"frames since accepted click {sec if sec is not None else '--'}", dim),
                ("", dim),
                ("left click    steer — every click draws the engine's pulse (a click", green),
                ("               within 150 ms of the last accepted one is ignored for the signal)", green),
                ("M / shift+M   next / previous selected model (same seed context, same", green),
                (f"               noise seed) — selected: {', '.join(self.cycle) or 'none'}", green),
                ("R             new random seed context      shift+R  identical rerun", green),
                ("space         pause                        .  or right  single step", green),
                ("F11 / alt+enter  fullscreen                Esc  leave fullscreen, then quit", green),
                ("Tab           compact HUD on/off           T  nearest / smooth scaling", green),
                ("S             screenshot                   H  this help", green),
                ("C             record clicks for replay     V  record the display to a GIF", green),
                ("Q             quit", green),
            ]
            pw = self._power_line()
            if pw:
                lines.append((pw, dim))
        else:
            lines = [(f"{self._model_tag()}   {fps:4.1f}/{TRAIN_HZ:.0f} fps"
                      + ("  DROPPING" if warn else "")
                      + f"   frame {self._gen_ms:4.1f} ms   held X {p.clicks.x:+.2f}"
                      + (f"   +{sec}f" if sec is not None else "   --")
                      + f"   {self.seeds.width}x{self.seeds.height}", red if warn else white)]
            pw = self._power_line()
            lines.append((f"seed {p.seed.label if p.seed else '?'}  noise {self.noise_seed}"
                          + (f"   {pw}" if pw else ""), dim))
        return lines

    def _draw_hud(self):
        import pygame
        if not self.show_hud and not self.show_help:
            return
        lines = self._hud_lines()
        if not lines:
            return
        pad = 5
        font = self.font_big if self.show_help else self.font
        rendered = [(font.render(t, True, c), c) for t, c in lines if t]
        w = max(s.get_width() for s, _ in rendered) + 2 * pad
        h = sum(s.get_height() for s, _ in rendered) + 2 * pad
        win_w, win_h = self.screen.get_size()
        if self.show_help:
            bar = pygame.Surface((win_w, win_h), pygame.SRCALPHA)
            bar.fill((0, 0, 0, 160))
            self.screen.blit(bar, (0, 0))
            x, y = max(pad, (win_w - min(w, win_w)) // 2), 10
        else:
            bar = pygame.Surface((win_w, h), pygame.SRCALPHA)
            bar.fill((0, 0, 0, 150))
            self.screen.blit(bar, (0, win_h - h))
            x, y = pad, win_h - h + pad
        for s, _ in rendered:
            self.screen.blit(s, (x, y))
            y += s.get_height()
        for i, (text, _) in enumerate(self.stat_lines):
            img = self.font.render(text, True, (255, 235, 150))
            self.screen.blit(img, (pad, win_h - h - (i + 1) * (img.get_height() + 2) - pad))

    # -- run --------------------------------------------------------------- #

    def run(self) -> int:
        import pygame
        try:
            self._open_window()
        except pygame.error as e:
            print(f"error: cannot open a window ({e}).\n"
                  f"  If this machine has no display, use a headless run instead:\n"
                  f"  .venv/bin/python play.py --model {self.model_name} --headless "
                  f"--frames 300 --out review/stage6/play_{self.model_name}.gif",
                  file=sys.stderr)
            pygame.quit()
            return 3
        self._player_for(self.model_name)
        self._reset_run(seed=self.seed_override)
        if self.args.record:
            self._recording_path = self.args.record
            self.recorder = SessionRecorder(self.model_name, self.player.checkpoint,
                                            self.player.seed, self.noise_seed)
        if self.args.video:
            self.display_rec = []
            self._display_rec_path = self.args.video
        clock = pygame.time.Clock()
        shown = False
        running = True
        try:
            if self.args.replay:
                self._start_replay()
            if self.args.warmup:
                # warm the GPU on throwaway frames, then restart the run so the timed
                # session (and the frame-time stats) start clean
                warm_seed = self.player.seed
                print(f"[play] warming up ({int(self.args.warmup)} frames) ...",
                      flush=True)
                for i in range(int(self.args.warmup)):
                    self.player.step(self.player.clicks.action, seed=self.noise_seed + i)
                self._reset_run(seed=warm_seed, noise_seed=self.noise_seed, keep_replay=True)
                self.player.session_frames = 0   # the warm-up is not part of the session
            if self.args.power:
                try:
                    # start sampling only now, so the ledger's fps covers the session
                    self.player.start_session()
                except Exception as e:
                    print(f"[play] live power readout unavailable: {e}", file=sys.stderr)

            print(self._status_text(), flush=True)
            print("[play] click in the window to steer. H for help, Esc/Q to quit.",
                  flush=True)
            t_start = time.perf_counter()
            while running:
                for ev in pygame.event.get():
                    if not self._handle_event(ev):
                        running = False
                self._update_pacing()
                now = time.perf_counter()
                if self._prune_pulses(now):
                    self._dirty = True
                if self._dirty or self.pulses:
                    self._draw(now)
                    self._dirty = False
                    if not shown:
                        shown = True
                        print(f"[play] window {self.screen.get_size()[0]}x"
                              f"{self.screen.get_size()[1]}"
                              f"{' fullscreen' if self.fullscreen else ''}, "
                              f"image {self._rect[2]}x{self._rect[3]} at "
                              f"({self._rect[0]},{self._rect[1]}), scale "
                              f"{self._scale:g}", flush=True)
                frame_ms = clock.tick(240)
                for i, item in enumerate(list(self.stat_lines)):
                    item[1] -= frame_ms / 1000.0
                    if item[1] <= 0:
                        self.stat_lines.remove(item)
                if self.args.exit_after and now - t_start >= self.args.exit_after:
                    print(f"[play] --exit-after {self.args.exit_after:g}s reached", flush=True)
                    running = False
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()
        return 0

    def _start_replay(self):
        """Load a recording and drive this run from its click log."""
        rep = Replay(self.args.replay)
        seed = self.seeds.get(rep.seed_context.get("episode"), rep.seed_context.get("start"))
        if seed is None:
            raise SystemExit(
                f"replay seed context {rep.seed_context.get('label')} is not in "
                f"{self.seeds.describe}; pass --packed to reach it")
        self._reset_run(seed=seed, noise_seed=rep.noise_seed, keep_replay=True)
        self.replay = rep
        self._set_status(f"replaying {os.path.basename(rep.path)} (from {rep.model})", 6.0)
        print(f"[play] replaying {rep.path}: {rep.n_frames} frames, {len(rep.clicks)} "
              f"clicks, recorded on {rep.model}; live input is ignored", flush=True)

    def _shutdown(self):
        if self.args.power:
            try:
                # stop sampling first, so the ledger's fps covers the session and not
                # the reporting and teardown below
                self.player.end_session(
                    run_id=f"play_{self.model_name}",
                    notes=f"player session, model {self.model_name}")
            except Exception as e:
                print(f"[play] could not write the energy ledger row: {e}", file=sys.stderr)
        if self.recorder is not None:
            path = self.recorder.save(self._recording_path)
            print(f"[play] recording saved: {path}", flush=True)
            self.recorder = None
        if self.display_rec is not None:
            frames, path = self.display_rec, self._display_rec_path
            self.display_rec = None
            if frames:
                write_gif(frames, path, scale=self._gif_scale(), fps=int(TRAIN_HZ))
                print(f"[play] display recording saved: {path} ({len(frames)} frames)",
                      flush=True)
        if self._lat:
            lat = np.asarray(self._lat)
            print(f"[play] frame time: mean {lat.mean():.1f} ms, p95 "
                  f"{np.percentile(lat, 95):.1f} ms, worst {lat.max():.1f} ms "
                  f"(budget {DT * 1000:.0f} ms); dropped steps {self._dropped}", flush=True)
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Headless
# --------------------------------------------------------------------------- #

def headless_run(args, player: Player, seed: Seed) -> int:
    """Generate frames without a window; write a GIF/strip and frame-time statistics."""
    player.reset(seed=seed)
    script = parse_clicks(args.clicks)
    n = int(args.frames)
    lat, frames = [], []
    dropped = 0
    print(f"[play] headless: model {player.name}, seed {player.seed.label}, "
          f"noise seed {args.noise_seed}, {n} frames at {TRAIN_HZ:.0f} Hz"
          f"{' (paced)' if args.pace else ''}", flush=True)
    # Warm the CUDA kernels / allocator first, then start the run over, so the
    # first frame's compilation cost does not look like a dropped step.
    if args.warmup:
        warm_seed = player.seed
        for i in range(int(args.warmup)):
            player.step(player.clicks.action, seed=args.noise_seed + i)
        player.reset(seed=warm_seed)
    t_next = time.perf_counter()
    for f in range(n):
        if args.pace:
            t_next += DT
            slack = t_next - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            elif slack < -2 * DT:
                dropped += 1
        t0 = time.perf_counter()
        for cf, cx in script:
            if cf == f:
                player.click(cx, f)
        frames.append(player.step(player.clicks.action, seed=args.noise_seed + f))
        lat.append((time.perf_counter() - t0) * 1000.0)
    lat_a = np.asarray(lat)
    stats = {
        "model": player.name, "checkpoint": registry.rel(player.checkpoint),
        "frames": n, "paced": bool(args.pace), "target_fps": TRAIN_HZ,
        "warmup_frames": int(args.warmup),
        "mean_ms": round(float(lat_a.mean()), 3),
        "p95_ms": round(float(np.percentile(lat_a, 95)), 3),
        "worst_ms": round(float(lat_a.max()), 3),
        "implied_fps": round(1000.0 / float(lat_a.mean()), 2),
        "budget_ms": round(DT * 1000.0, 1),
        "dropped_steps": dropped,
        "kept_up": bool(lat_a.max() < DT * 1000.0),
        "seed_context": player.seed.label, "noise_seed": int(args.noise_seed),
        "resolution": [player.seeds.width, player.seeds.height],
        "click_script": args.clicks,
    }
    print(f"[play] frame time: mean {stats['mean_ms']} ms, p95 {stats['p95_ms']} ms, "
          f"worst {stats['worst_ms']} ms -> {stats['implied_fps']} fps "
          f"(budget {stats['budget_ms']} ms); dropped steps {dropped}; "
          f"kept up: {stats['kept_up']}", flush=True)
    if args.out:
        write_gif(frames, args.out, scale=max(1, args.gif_scale), fps=int(args.gif_fps))
        print(f"[play] wrote {args.out}", flush=True)
    if args.strip:
        write_strip(frames, args.strip)
        print(f"[play] wrote {args.strip}", flush=True)
    if args.record:
        rec = SessionRecorder(player.name, player.checkpoint, player.seed,
                             int(args.noise_seed), notes="headless --clicks session")
        for cf, cx in script:
            rec.click(cf, cx)
        rec.advance(n)
        path = rec.save(args.record)
        print(f"[play] recorded {len(script)} clicks of a {n}-frame session -> {path}",
              flush=True)
    if args.report_frames:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_frames)), exist_ok=True)
        with open(args.report_frames, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=1)
        print(f"[play] wrote {args.report_frames}", flush=True)
    return 0


def replay_headless(args, player: Player, rep: Replay) -> int:
    """Replay a recorded session against ``player`` and write the result."""
    seed = player.seeds.get(rep.seed_context.get("episode"), rep.seed_context.get("start"))
    if seed is None:
        raise SystemExit(f"replay seed context {rep.seed_context.get('label')} is not in "
                         f"{player.seeds.describe}; pass --packed to reach it")
    player.reset(seed=seed)
    n = int(args.frames) if args.frames_given else rep.n_frames
    frames = player.generate(n, rep.clicks, noise_seed=rep.noise_seed)
    print(f"[play] replayed {len(frames)} frames of {os.path.basename(rep.path)} "
          f"(recorded on {rep.model}) with model {player.name}, seed {player.seed.label}",
          flush=True)
    if args.out:
        write_gif(frames, args.out, scale=max(1, args.gif_scale), fps=int(args.gif_fps))
        print(f"[play] wrote {args.out}", flush=True)
    if args.strip:
        write_strip(frames, args.strip)
        print(f"[play] wrote {args.strip}", flush=True)
    if args.report_frames:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_frames)), exist_ok=True)
        with open(args.report_frames, "w", encoding="utf-8") as fh:
            json.dump({"model": player.name, "frames": len(frames),
                       "replay": registry.rel(rep.path),
                       "seed_context": player.seed.label,
                       "noise_seed": rep.noise_seed}, fh, indent=1)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_clicks(spec: str):
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        f, x = part.split(":")
        out.append((int(f), float(x)))
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="play.py",
        description="NeuralGameEngine player: run a registered model in real time at "
                    "20 Hz and steer it with clicks.",
        epilog=(
            "models are named after their internal training run (flow96, flow96c, ...);\n"
            "the old 5a/5b letters still work as aliases (5a=flow96, 5b=flow96c).\n"
            "M cycles only the *selected* models; any name or run directory is playable\n"
            "explicitly with --model, including candidates that were not selected.\n"
            "\n"
            "examples:\n"
            "  ./play.sh                                        newest selected model, windowed\n"
            "  ./play.sh --model flow96                         any model, by internal name\n"
            "  ./play.sh --model flow96c --fullscreen           start fullscreen\n"
            "  ./play.sh --model flow96e                        an unregistered run (runs/flow96e)\n"
            "  ./play.sh --list-models                          what is registered\n"
            "  .venv/bin/python play.py --model flow96 --headless --frames 300 \\\n"
            "      --clicks \"10:0.8,90:-0.8\" --out review/stage6/play_flow96.gif\n"
            "  .venv/bin/python play.py --model flow96 --replay models/recordings/<f>.json\n"
            "\n"
            "keys:\n"
            "  left click     steer; every click shows the engine's pulse (a click\n"
            "                 within 150 ms of the last accepted one does not move it)\n"
            "  F11/alt+enter  fullscreen        Esc  leave fullscreen, then quit\n"
            "  M / shift+M    next / previous SELECTED model (same seed context, same\n"
            "                 noise seed, and the same recorded clicks during a replay)\n"
            "  R              restart with a new random seed context and noise seed\n"
            "  shift+R        restart with the identical seed context and noise seed\n"
            "  space          pause             . or right arrow  single step\n"
            "  Tab            compact HUD       H or F1  full help\n"
            "  T              nearest / smooth scaling     S or F2  screenshot\n"
            "  C              record the session (clicks) for replay\n"
            "  V              record the display to a GIF\n"
            "  P              print a one-line status to the terminal\n"
            "  Q              quit\n"
            "\n"
            "where things live:\n"
            "  models/registry.json        the registry (name -> checkpoint, lineage, evals)\n"
            "  models/<name>/generator.pt  the frozen checkpoint copy for a model\n"
            "  models/seeds.npz            cached seed contexts (build_seeds.py)\n"
            "  models/recordings/          recorded sessions for replay\n"
            "  review/stage6/              screenshots and display recordings\n"
            "add a model with: .venv/bin/python register_model.py --run flow96e \\\n"
            "    --core runs/steer96/best.pt --description \"...\" [--not-selected]"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("model")
    g.add_argument("--model", default=None,
                   help="a model name — its internal run name (flow96c), an alias (5b), "
                        "or a run directory name (flow96e) — default: newest selected")
    g.add_argument("--list-models", action="store_true",
                   help="list registered models (and which are selected) and exit")
    g.add_argument("--checkpoint", default=None,
                   help="run this checkpoint directly, bypassing the registry "
                        "(requires --packed for seeds)")
    g.add_argument("--device", default=None, help="torch device (default: auto)")
    g.add_argument("--hist-extra", type=int, default=8,
                   help="real click-history rows kept before the context window")
    s = ap.add_argument_group("seed context")
    s.add_argument("--packed", default=None,
                   help="packed dataset to take seed frames from (default: the seed cache)")
    s.add_argument("--seeds", default=registry.DEFAULT_SEEDS, help="seed-context cache")
    s.add_argument("--split", default="val", help="split to draw random seeds from")
    s.add_argument("--seed-context", dest="seed_context", default=None,
                   help="exact seed clip as EPISODE:START")
    s.add_argument("--noise-corr", dest="noise_corr", type=float, default=0.0,
                   help="AR(1) correlation of the flow noise across frames: 0.0 (the "
                        "default) is fresh independent noise every frame, as before; "
                        "try 0.9 to stabilise a rollout")
    s.add_argument("--noise-seed", dest="noise_seed", type=int, default=0,
                   help="base seed for the generator noise (frame f uses noise_seed+f)")
    w = ap.add_argument_group("window")
    w.add_argument("--scale", type=int, default=None,
                   help="initial integer pixel scale (default: ~40%% of the desktop "
                        "width, at most 16)")
    w.add_argument("--window-w", type=int, default=None, help="initial window width")
    w.add_argument("--window-h", type=int, default=None, help="initial window height")
    w.add_argument("--fullscreen", action="store_true", help="start fullscreen")
    w.add_argument("--smooth", action="store_true",
                   help="start with smooth (not nearest) scaling")
    w.add_argument("--no-hud", dest="no_hud", action="store_true", help="start with the HUD off")
    w.add_argument("--help-overlay", action="store_true",
                   help="start with the help overlay shown")
    w.add_argument("--power", action="store_true", help="sample GPU power during the session")
    w.add_argument("--no-power", dest="power", action="store_false",
                   help="do not sample power")
    w.add_argument("--exit-after", type=float, default=0.0,
                   help="quit automatically after this many seconds (0 = run until you quit)")
    h = ap.add_argument_group("headless")
    h.add_argument("--headless", action="store_true", help="no window; generate and write files")
    h.add_argument("--frames", type=int, default=300, help="frames to generate")
    h.add_argument("--clicks", default="", help='scripted clicks, "frame:x,frame:x"')
    h.add_argument("--pace", action="store_true",
                   help="headless: hold the real 20 Hz rate instead of running flat out")
    h.add_argument("--warmup", type=int, default=5,
                   help="frames generated (then discarded) before timing, to warm the GPU")
    h.add_argument("--out", default=None, help="GIF to write (headless / replay)")
    h.add_argument("--strip", default=None, help="frame-strip PNG to write (headless)")
    h.add_argument("--gif-scale", type=int, default=4, help="GIF upscale factor")
    h.add_argument("--gif-fps", type=int, default=20, help="GIF playback rate")
    h.add_argument("--report-frames", default=None,
                   help="write per-frame time statistics as JSON here")
    r = ap.add_argument_group("recording")
    r.add_argument("--record", default=None, help="record this session to this file")
    r.add_argument("--replay", default=None, help="replay a recorded session")
    r.add_argument("--video", default=None,
                   help="record the display to this GIF for the whole session")
    ap.set_defaults(power=True)
    return ap


def _pick_window_size(args, seeds: SeedBank) -> tuple[int, int]:
    """A sensible, reasonably large default window (the old player's was tiny)."""
    if args.window_w and args.window_h:
        return int(args.window_w), int(args.window_h)
    scale = args.scale
    if scale is None:
        dw, dh = 1920, 1080
        try:
            import pygame
            pygame.display.init()
            info = pygame.display.Info()
            dw, dh = info.current_w or dw, info.current_h or dh
            pygame.display.quit()
        except Exception:
            pass
        scale = max(1, min(dw * 4 // 5 // max(1, seeds.width),
                           dh * 4 // 5 // max(1, seeds.height), 16))
    return max(1, seeds.width * int(scale)), max(1, seeds.height * int(scale))


def resolve_start(catalog: Catalog, args) -> tuple[str, str]:
    """``(name, checkpoint path)`` for the requested model, or a raw ``--checkpoint``."""
    if args.checkpoint:
        if not args.packed:
            raise SystemExit("error: --checkpoint bypasses the registry, so --packed is "
                             "required (the seed cache is indexed by registered seeds)")
        if not os.path.exists(args.checkpoint):
            raise SystemExit(f"error: checkpoint not found: {args.checkpoint}")
        name = os.path.basename(os.path.dirname(os.path.abspath(args.checkpoint))) or "raw"
        return name, args.checkpoint
    if not catalog.known():
        raise SystemExit(
            "error: no models are registered yet.\n"
            "  register one (a copy of its checkpoint is frozen first):\n"
            "    .venv/bin/python register_model.py --run flow96c \\\n"
            "        --core runs/steer96/best.pt --description \"flow96c: overnight winner\"\n"
            "  then: ./play.sh --model flow96c")
    name = args.model or catalog.default()
    info = catalog.info(name)              # raises with the known names if unknown
    if info["kind"] == "source":
        print(f"[play] WARNING: {info['name']} has no frozen copy; loading "
              f"{info['entry']['checkpoint_source']} (a later training run could "
              f"overwrite what {info['name']} means)", file=sys.stderr)
    elif info["kind"] == "run":
        print(f"[play] note: {info['name']} is not in the registry; loading "
              f"{registry.rel(info['path'])} directly", file=sys.stderr)
    return info["name"], info["path"]


def main(argv=None) -> int:
    ap = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = ap.parse_args(argv)
    args.frames_given = any(a == "--frames" or a.startswith("--frames=")
                            for a in argv_list)

    if args.list_models:
        print(registry.format_table())
        return 0

    catalog = Catalog()
    try:
        name, ckpt = resolve_start(catalog, args)
    except registry.RegistryError as e:
        print(f"error: {e}\n  list them with: ./play.sh --list-models", file=sys.stderr)
        return 2
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 2
    args.model = name

    seeds = open_seed_bank(args, catalog.reg)
    if not args.headless:
        args.window_w, args.window_h = _pick_window_size(args, seeds)

    seed_override = None
    if args.seed_context:
        try:
            ep, st = (int(v) for v in args.seed_context.split(":"))
        except ValueError:
            print(f"error: --seed-context should be EPISODE:START, got "
                  f"{args.seed_context!r}", file=sys.stderr)
            return 2
        seed_override = seeds.get(ep, st)
        if seed_override is None:
            print(f"error: seed context {ep}:{st} is not in {seeds.describe}; "
                  f"pass --packed to use it directly", file=sys.stderr)
            return 2

    player = Player(ckpt, seeds, device=args.device, hist_extra=args.hist_extra,
                    name=name, split=args.split,
                    noise_corr=float(getattr(args, "noise_corr", 0.0)))

    if args.headless:
        if args.replay:
            try:
                rep = Replay(args.replay)
            except (OSError, ValueError, SystemExit) as e:
                print(f"error: cannot read the replay {args.replay}: {e}", file=sys.stderr)
                return 2
            return replay_headless(args, player, rep)
        seed = seed_override or seeds.pick(player.rng, args.split)
        return headless_run(args, player, seed)

    client = Client(args, catalog, catalog.cycle(), name, seeds)
    client.seed_override = seed_override
    return client.run()


if __name__ == "__main__":
    raise SystemExit(main())
