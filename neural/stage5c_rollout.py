"""Stage 5c — the single evaluation path and the 4.0d health numbers.

Everything that makes a decision this night goes through here, and everything here
follows the *live player's own maths* (``play.Player.step``): fresh independent noise
per frame, the real click history and debounce state restored from the seed, the real
seed contexts from ``models/seeds.npz``. The Stage 5b scorecard did **not** do that (it
holds one noise tensor for a whole rollout — see ``STAGE5C-PROGRESS.md`` §4.0a), which
is why its ``frame_change`` ordering was the opposite of what the user sees.

Nothing here changes any default of any existing model; it only reads.

The four health numbers (4.0d), all on 300-step rollouts unless said otherwise, over
*generated* frames only:

1. ``frame_change``  — mean |Δ| grey levels between consecutive frames, steps 100-300.
2. ``hf_ratio``      — high-frequency (4-neighbour Laplacian) energy against real
                       consecutive frames, same display scale.
3. ``luma_mean``, ``contrast`` — mean grey and its standard deviation, steps 200-300
                       (catches collapse to uniform / saturated / purple frames).
4. ``near_black``    — share of steps 200-300 whose mean grey < ``BLACK_LUMA``.

Real references come from the packed set's val split, displayed the same way.
"""

from __future__ import annotations

import numpy as np
import torch

from dataset import frames_to_tensor
from eval_harness import features_from_history, model_predict

# --- thresholds (documented so the bands are reproducible) ------------------- #
BLACK_LUMA = 20.0          # a frame whose mean grey is below this is "near black"
FRAME_CHANGE_RANGE = (100, 300)   # generated-frame indices for metric 1
COLLAPSE_RANGE = (200, 300)       # generated-frame indices for metrics 3 and 4

GREY = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def to_grey(frames: np.ndarray) -> np.ndarray:
    """uint8 [N,H,W,3] -> float32 [N,H,W] grey in 0..255."""
    return (frames.astype(np.float32) @ GREY)


def laplacian_energy(grey: np.ndarray) -> float:
    """Mean |discrete Laplacian| over a stack of grey frames [N,H,W]."""
    lap = (4.0 * grey[:, 1:-1, 1:-1] - grey[:, :-2, 1:-1] - grey[:, 2:, 1:-1]
           - grey[:, 1:-1, :-2] - grey[:, 1:-1, 2:])
    return float(np.abs(lap).mean())


def health(frames: np.ndarray, real_hf: float | None = None) -> dict:
    """The 4.0d health numbers for a rollout of uint8 frames [N,H,W,3]."""
    g = to_grey(frames)
    lo, hi = FRAME_CHANGE_RANGE
    lo = min(lo, max(0, len(g) - 2))
    hi = min(hi, len(g))
    d = np.abs(np.diff(g[lo:hi], axis=0))
    c_lo, c_hi = COLLAPSE_RANGE
    c_lo = min(c_lo, max(0, len(g) - 1))
    c_hi = min(c_hi, len(g))
    late = g[c_lo:c_hi]
    out = {
        "n_frames": int(len(g)),
        "frame_change": float(d.mean()) if d.size else None,
        "luma_mean": float(late.mean()) if late.size else None,
        "contrast": float(late.std()) if late.size else None,
        "near_black": float((late.mean(axis=(1, 2)) < BLACK_LUMA).mean())
        if late.size else None,
        "hf_energy": laplacian_energy(g),
    }
    if real_hf:
        out["hf_ratio"] = out["hf_energy"] / real_hf
    return out


# The pass band, calibrated in Phase 0 against the user's known verdicts
# (flow96 healthy; flow96c and flow96d not). See STAGE5C-PROGRESS.md §4.0d.
PASS_BAND = {
    "frame_change": (3.0, 22.0),
    "hf_ratio": (0.30, 1.50),
    "contrast": (30.0, 62.0),
    "luma_mean": (50.0, 170.0),
    "near_black": (0.0, 0.15),
}


def in_band(h: dict, band: dict | None = None) -> dict:
    """Which health keys are inside the pass band (and whether all of them are)."""
    band = band or PASS_BAND
    checks = {}
    for k, (lo, hi) in band.items():
        v = h.get(k)
        checks[k] = bool(v is not None and lo <= float(v) <= hi)
    checks["all"] = all(checks.values())
    return checks


def real_reference(packed_path: str, n_clips: int = 8, steps: int = 300,
                   split: str = "val", seed: int = 0) -> dict:
    """Real consecutive frames from long episodes, as an array + its health numbers."""
    from dataset import PackedDataset
    packed = PackedDataset(packed_path, in_memory=False)
    rng = np.random.default_rng(seed)
    eps = [e for e in packed.episodes if e["split"] == split and e["length"] >= steps]
    idx = rng.permutation(len(eps))[:n_clips]
    clips = [np.asarray(packed.images[eps[int(i)]["frame_offset"]:
                                      eps[int(i)]["frame_offset"] + steps]) for i in idx]
    arr = np.stack(clips)                                    # [n,steps,H,W,3]
    flat = arr.reshape(-1, *arr.shape[2:])
    hf = laplacian_energy(to_grey(flat))
    stats = health(flat, real_hf=hf)
    stats["hf_energy"] = hf
    return {"frames": arr, "stats": stats, "n_clips": len(clips)}


# --------------------------------------------------------------------------- #
# Live-path rollouts
# --------------------------------------------------------------------------- #

def open_player(model: str, device: str | None = None, seeds_path: str | None = None,
                **kw):
    """A ``play.Player`` for ``model`` (a registry name, a bare run name, or a .pt path)."""
    import os
    import play
    if model.endswith(".pt") and os.path.exists(model):
        path, name = model, os.path.basename(os.path.dirname(os.path.abspath(model)))
    else:
        info = play.Catalog().info(model)
        path, name = info["path"], info["name"]
    seeds = play.CacheSeeds(seeds_path or "models/seeds.npz")
    return play.Player(path, seeds, device=device, name=name, **kw)


def pick_seeds(n: int = 3, split: str = "val", offset: int = 0):
    """``n`` deterministic seed contexts from the cache (stable across models)."""
    import play
    seeds = play.CacheSeeds("models/seeds.npz")
    idx = [i for i in range(len(seeds.episode)) if str(seeds.split[i]) == split]
    return [seeds._make(int(idx[(offset + j) % len(idx)])) for j in range(n)]


class Drive:
    """The closed loop of ``play.Player.step``, usable with an in-memory model.

    ``Player`` itself is preferred when a checkpoint path is involved; this exists so
    the training driver can health-check the EMA weights without a second model load.
    The maths is identical (asserted in ``tests/test_stage5c.py``).
    """

    def __init__(self, model, seed, device="cuda", K: int = 4, hist_extra: int = 8):
        import play
        self.model = model
        self.device = device
        self.K = int(K)
        self.hist_extra = int(hist_extra)
        self.frames = frames_to_tensor(np.asarray(seed["frames"])).to(device)
        self.hist = [np.asarray(r, dtype=np.float32) for r in seed["hist"]]
        self.clicks = play.ClickState()
        self.clicks.restore(np.stack(self.hist))
        self.frame_index = 0
        self._trim()

    def _trim(self):
        keep = self.hist_extra + self.K
        if len(self.hist) > keep:
            del self.hist[:len(self.hist) - keep]

    def _predict(self, ctx, a, feats, seed, noise, guidance, level):
        """Direct predict when a knob is used, else the shared ``model_predict``."""
        if noise is None and guidance == 1.0:
            return model_predict(self.model, ctx, a, feats, seed)
        kw = {"features": feats, "seed": seed}
        params = self._predict_params()
        if "noise" in params:
            kw["noise"] = noise
        if "guidance" in params:
            kw["guidance"] = guidance
        if "level" in params:
            kw["level"] = level
        return self.model.predict(ctx, a, **kw)

    def _predict_params(self):
        if not hasattr(self, "_pp"):
            import inspect
            try:
                self._pp = inspect.signature(self.model.predict).parameters
            except (TypeError, ValueError):                # pragma: no cover
                self._pp = {}
        return self._pp

    def step(self, seed: int, noise: torch.Tensor | None = None,
             guidance: float = 1.0, level: int | None = None):
        action = self.clicks.action
        self.hist.append(np.array([action[0], action[1]], dtype=np.float32))
        self._trim()
        feats = features_from_history(np.stack(self.hist), self.K, self.device)
        ctx = self.frames[-self.K:].unsqueeze(0)
        a = torch.from_numpy(np.ascontiguousarray(
            np.stack(self.hist[-self.K:])))[None].to(self.device)
        out = self._predict(ctx, a, feats, seed, noise, guidance, level)
        nxt = out[0] if isinstance(out, tuple) else out
        self.frames = torch.cat([self.frames, nxt], dim=0)
        self.frame_index += 1
        return nxt[0]

    def generate(self, steps: int, clicks=None, noise_seed: int = 0,
                 rho: float = 0.0, guidance: float = 1.0,
                 level: int | None = None) -> np.ndarray:
        """uint8 [steps,H,W,3]; ``rho=0`` is the player's fresh-noise behaviour."""
        import play
        script: dict[int, list[float]] = {}
        for cf, cx in (clicks or []):
            script.setdefault(int(cf), []).append(float(cx))
        H, W = self.frames.shape[-2:]
        z = None
        out_frames = []
        with torch.no_grad():
            for f in range(steps):
                for cx in script.get(f, ()):
                    self.clicks.click(cx, f)
                noise = None
                if rho > 0.0:
                    g = torch.Generator(device="cpu").manual_seed(int(noise_seed) + f)
                    fresh = torch.randn((1, 3, H, W), generator=g)
                    z = fresh if z is None else rho * z + (1.0 - rho * rho) ** 0.5 * fresh
                    noise = z
                out_frames.append(play._to_uint8(
                    self.step(int(noise_seed) + f, noise, guidance=guidance,
                              level=level)))
        return np.stack(out_frames)


def rollout_model(model, seed, steps: int = 300, device: str = "cuda", clicks=None,
                  noise_seed: int = 0, rho: float = 0.0, K: int = 4,
                  guidance: float = 1.0) -> np.ndarray:
    """In-memory-model equivalent of ``rollout_live`` (used by the training driver)."""
    was_training = model.training
    model.eval()
    try:
        return Drive(model, seed, device=device, K=K).generate(
            steps, clicks=clicks, noise_seed=noise_seed, rho=rho, guidance=guidance)
    finally:
        if was_training:
            model.train()


def rollout_live(player, seed, steps: int = 300, clicks=None,
                 noise_seed: int = 0, rho: float = 0.0) -> np.ndarray:
    """One rollout through ``play.Player`` itself -> uint8 [steps,H,W,3].

    ``Player.generate`` verbatim for ``rho=0`` (fresh noise ``noise_seed + f`` every
    frame, real click history, debounce restored from the seed).
    """
    import play
    if rho <= 0.0:
        player.reset(seed=seed)
        frames = player.generate(steps, click_script=clicks, noise_seed=noise_seed)
        return np.stack([play._to_uint8(f) for f in frames])
    return Drive(player.model, seed, device=player.device, K=player.K,
                 hist_extra=player.hist_extra).generate(
        steps, clicks=clicks, noise_seed=noise_seed, rho=rho)


def contact_sheet(frames: np.ndarray, path: str, steps, scale: int = 3):
    """A row of the frames at the given *generated* indices (1-based like a step)."""
    from PIL import Image
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    imgs = []
    for s in steps:
        i = min(max(0, int(s) - 1), len(frames) - 1)
        u = frames[i]
        imgs.append(Image.fromarray(u).resize((u.shape[1] * scale, u.shape[0] * scale),
                                              Image.Resampling.NEAREST))
    w, h = imgs[0].size
    sheet = Image.new("RGB", (len(imgs) * (w + 4), h + 4), (20, 20, 20))
    for i, im in enumerate(imgs):
        sheet.paste(im, (i * (w + 4), 2))
    sheet.save(path)
    return path
