"""Frozen judges for Stage 5b (Section 4 of the overnight brief).

Nothing in this module is importable from the playable inference path: the
judges exist only for evaluation and are allowed to use sidecar labels (the
opening detector is trained on REAL frames only, with labels derived from the
sidecar's doorway candidates, and is never retrained on generated frames).

Judges implemented here
-----------------------
  R. ``radial_psd`` / ``spectrum_report`` — radially averaged power spectrum,
     per-band ratios and a scalar distance; ``laplacian_var`` and
     ``multiscale_gradient`` for the grain-vs-blur distinction.
  D. ``OpeningDetector`` — a small CNN that answers "is a doorway or side
     opening visible in this frame?".  Labels come from the sidecar's per-frame
     ``candidates`` list (a doorway projected inside the viewport = positive).
     Frames with NO candidate in view are the negatives, and the subset of those
     that are mostly dark (heavy fog / the black far wall of a big room) is kept
     as an explicit NEGATIVE CONTROL whose precision is reported separately.
  C. ``train_realfake`` — a fresh small CNN per candidate that has to tell real
     frames from generated ones; held-out AUC at each horizon (0.5 = ideal).
  H. ``health_stats`` — colour, dark-region colour, luminance drift, near-black
     and near-white shares, frame-to-frame change, frozen-image check.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# small shared bits
# --------------------------------------------------------------------------- #

def to_float01(u8: np.ndarray) -> np.ndarray:
    """(N,H,W,C) uint8 -> (N,H,W,C) float32 in [0,1]."""
    a = np.asarray(u8)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    return np.clip(a.astype(np.float32), 0.0, 1.0)


def luminance(f01: np.ndarray) -> np.ndarray:
    """(N,H,W,3) -> (N,H,W) Rec.601 luminance."""
    return (0.299 * f01[..., 0] + 0.587 * f01[..., 1] + 0.114 * f01[..., 2]).astype(np.float32)


# --------------------------------------------------------------------------- #
# R. realness: spectrum and gradient statistics
# --------------------------------------------------------------------------- #

def radial_psd(frames_u8: np.ndarray, n_bins: int = 24, eps: float = 1e-8):
    """Radially averaged power spectrum of a stack of frames.

    Frames are converted to luminance, mean-removed, Hann-windowed and FFTed; the
    power is averaged over frames and binned by spatial frequency.  Returns
    ``(freq, power, count)`` where ``freq`` is in cycles/pixel (0..0.5).

    The radial average is what makes this robust to *where* structure sits: a
    blurry model loses high frequencies, a grainy one gains them, and neither can
    be hidden by matching the mean gradient.
    """
    f = luminance(to_float01(frames_u8))
    n, h, w = f.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    acc = np.zeros((h, w), dtype=np.float64)
    for i in range(n):
        x = (f[i] - f[i].mean()) * win
        acc += np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    acc /= max(1, n)
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    rmax = r.max()
    idx = np.clip((r / rmax * (n_bins - 1)).round().astype(np.int32), 0, n_bins - 1)
    power = np.bincount(idx.ravel(), weights=acc.ravel(), minlength=n_bins)
    count = np.bincount(idx.ravel(), minlength=n_bins).astype(np.float64)
    freqs = (np.arange(n_bins) / (n_bins - 1)) * rmax
    power = power / np.maximum(count, 1)
    return freqs, power + eps, count


def spectrum_report(gen_u8: np.ndarray, real_u8: np.ndarray, n_bins: int = 24) -> dict:
    """Spectrum distance and per-band power ratios between generated and real."""
    fg, pg, _ = radial_psd(gen_u8, n_bins)
    fr, pr, _ = radial_psd(real_u8, n_bins)
    ratio = pg / pr
    # distance on log-power, weighted so low frequencies (which carry most of the
    # energy and dominate the visual impression) don't swamp the high ones.
    d = np.sqrt(np.mean((np.log(ratio)) ** 2))
    bands = {}
    edges = [(0.0, 0.08), (0.08, 0.16), (0.16, 0.25), (0.25, 0.35), (0.35, 0.51)]
    for lo, hi in edges:
        m = (fr >= lo) & (fr < hi)
        if m.sum():
            bands[f"{lo:.2f}-{hi:.2f}"] = float(np.mean(ratio[m]))
    return {
        "spectrum_distance": float(d),
        "band_ratio": bands,
        "freqs": [round(float(x), 4) for x in fr],
        "gen_psd": [float(x) for x in pg],
        "real_psd": [float(x) for x in pr],
    }


def _laplacian(f01: np.ndarray) -> np.ndarray:
    """3x3 discrete Laplacian on the luminance of a frame stack."""
    f = luminance(f01)
    t = torch.from_numpy(f)[:, None]
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    out = F.conv2d(t, k, padding=1)
    return out[:, 0].numpy()


def laplacian_var(frames_u8: np.ndarray) -> float:
    return float(np.var(_laplacian(to_float01(frames_u8))))


def multiscale_gradient(frames_u8: np.ndarray, scales=(1, 2, 4)) -> dict:
    """Mean |gradient| at several blur levels (a grain-vs-structure discriminator:
    grain survives scale 1 only, real structure survives all scales)."""
    f = to_float01(frames_u8)
    t = torch.from_numpy(luminance(f))[:, None]
    sx = torch.tensor([[-1.0, 0.0, 1.0]]).view(1, 1, 1, 3) / 2.0
    sy = torch.tensor([[-1.0], [0.0], [1.0]]).view(1, 1, 3, 1) / 2.0
    out = {}
    cur = t
    for s in scales:
        if s > 1:
            cur = F.avg_pool2d(cur, 2, 2)
        gx = F.conv2d(cur, sx, padding=(0, 1))
        gy = F.conv2d(cur, sy, padding=(1, 0))
        g = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
        out[f"grad_scale{int(s)}"] = float(g.mean())
    return out


# --------------------------------------------------------------------------- #
# D. the frozen opening detector
# --------------------------------------------------------------------------- #

VISIBLE_X = 0.98       # a candidate is "in view" when |screenX| is inside this
OPEN_T = 0.35          # "prominently in view": reachable, close in screen space

# Why the label looks like this (measured on 34,744 val frames before freezing):
# a doorway candidate is in view in 97.4% of frames and the nearest one is within
# |screenX| <= 0.3 in 91%, so a plain "is any door on screen?" question is true
# almost always and cannot be discriminative.  Requiring the candidate to be
# *reachable* and close to straight ahead (both screenDistance and |screenX| within
# 0.35) gives a 54.3% base rate, and the 2.6% of frames with no candidate in view
# are the negative controls the brief asks for (heavy fog / the black far wall of a
# large room).


def opening_labels(sidecar, packed, split="train", max_frames=None, seed=0):
    """Build (episode, frame, label, is_no_door_control) rows from sidecar candidates.

    POSITIVE: a reachable doorway candidate with ``screenDistance <= OPEN_T`` and
    ``|screenX| <= OPEN_T`` — a walkable opening prominently in view.
    NEGATIVE: everything else.
    CONTROL:  no candidate projects inside ``|screenX| <= VISIBLE_X`` — the frames
    with nothing door-like on screen at all (fog, or a large room whose far wall is
    black).  These are the frames the detector must not fire on.
    """
    rows = []
    eps = packed.episodes_by_split[split]
    rng = np.random.RandomState(seed)
    for e in eps:
        off, length = e["frame_offset"], e["length"]
        cands = sidecar.candidates[off: off + length]
        for i, cand in enumerate(cands):
            if cand is None:
                continue
            if not len(cand):
                rows.append((e["episode_index"], i, 0, 1))
                continue
            vis = [c for c in cand if abs(float(c["screenX"])) <= VISIBLE_X]
            if not vis:
                rows.append((e["episode_index"], i, 0, 1))
                continue
            near = [c for c in vis if bool(c.get("reachable"))] or vis
            md = min(float(c["screenDistance"]) for c in near)
            mx = min(abs(float(c["screenX"])) for c in near)
            lab = 1 if (md <= OPEN_T and mx <= OPEN_T) else 0
            rows.append((e["episode_index"], i, lab, 0))
    if max_frames:
        idx = rng.choice(len(rows), size=min(max_frames, len(rows)), replace=False)
        rows = [rows[i] for i in idx]
    return rows


def _opening_targets(rows, packed):
    """Split rows into positives and negatives (with dark-control flags)."""
    pos = [(e, i) for (e, i, lab, _) in rows if lab == 1]
    neg = [(e, i) for (e, i, lab, _) in rows if lab == 0]
    return pos, neg


class _SmallCNN(nn.Module):
    """Small conv net; ``instance_norm`` removes each frame's own mean and gain.

    The opening detector uses it so that overall darkness (fog) cannot be used as a
    shortcut for "no doorway": measured before the fix, the detector fired on 94% of
    the darkest no-door frames while firing on only 19% of no-door frames overall.
    """

    def __init__(self, in_ch=3, width=16, dropout=0.1, instance_norm=False):
        super().__init__()
        self.instance_norm = instance_norm
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(w * 2, w * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(w * 2, w * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(w * 4, 1))

    def forward(self, x):
        if self.instance_norm:
            m = x.mean(dim=(1, 2, 3), keepdim=True)
            v = x.var(dim=(1, 2, 3), keepdim=True)
            x = (x - m) / torch.sqrt(v + 1e-6)
        return self.head(self.net(x)).squeeze(-1)


class OpeningDetector:
    """Frozen 'is a doorway/opening visible?' judge.

    Trained once on REAL frames.  Never trained on generated frames (the brief
    forbids it), so its output on generated frames is an out-of-distribution
    readout — which is exactly what makes it evidence about structure.
    """

    def __init__(self, model=None, thresh=0.5, meta=None):
        self.model = model
        self.thresh = thresh
        self.meta = meta or {}

    def save(self, path):
        torch.save({"state": self.model.state_dict(), "thresh": self.thresh, "meta": self.meta}, path)

    @classmethod
    def load(cls, path, device="cuda"):
        ck = torch.load(path, map_location=device, weights_only=False)
        m = _SmallCNN(instance_norm=bool((ck.get("meta") or {}).get("instance_norm")))
        m.load_state_dict(ck["state"])
        m.to(device).eval()
        return cls(m, ck.get("thresh", 0.5), ck.get("meta"))

    @torch.no_grad()
    def prob(self, frames_u8, device="cuda", batch=256) -> np.ndarray:
        """P(an opening is visible) for each frame."""
        x = torch.from_numpy(to_float01(frames_u8)).permute(0, 3, 1, 2).contiguous()
        probs = []
        for i in range(0, x.shape[0], batch):
            out = self.model(x[i: i + batch].to(device))
            probs.append(torch.sigmoid(out).float().cpu().numpy())
        return np.concatenate(probs) if probs else np.zeros(0, dtype=np.float32)

    def positive(self, frames_u8, device="cuda") -> np.ndarray:
        return self.prob(frames_u8, device) >= self.thresh


def train_opening_detector(packed, sidecar, steps=700, device="cuda", seed=0, log=None,
                           out_path=None, width=16):
    """Train and validate the opening detector on REAL frames only.

    No-door-control frames are oversampled 3x so the net is explicitly taught the
    hard case (black fog / a large dark room with no opening), and the input is
    instance-normalised so overall darkness is not a usable cue.
    """
    torch.manual_seed(seed)
    rows = opening_labels(sidecar, packed, "train", max_frames=80000, seed=seed)
    pos = [(e, i) for (e, i, lab, _) in rows if lab == 1]
    neg = [(e, i) for (e, i, lab, c) in rows if lab == 0 and not c]
    ctl = [(e, i) for (e, i, lab, c) in rows if c]

    def _imgs(items):
        return np.stack([np.asarray(packed.images[packed.episodes[e]["frame_offset"] + f])
                         for e, f in items])

    rng = np.random.RandomState(seed)
    n = 6000
    pick = lambda lst, k: [lst[i] for i in rng.choice(len(lst), min(k, len(lst)), replace=False)]
    Xp, Xn = _imgs(pick(pos, n)), _imgs(pick(neg, n))
    Xc = _imgs(pick(ctl, max(200, len(ctl) // 6)))
    # 3x oversampling of the no-door controls (repeat, then deduplicate by shuffle)
    X = np.concatenate([Xp, Xn, np.concatenate([Xc] * 3)])
    y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn) + 3 * len(Xc))]).astype(np.float32)

    meta = {"instance_norm": True, "open_t": OPEN_T, "width": width,
            "train_pos": int(len(Xp)), "train_neg": int(len(Xn)), "train_control": int(len(Xc)),
            "steps": steps}
    model = _SmallCNN(width=width, instance_norm=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    xt = torch.from_numpy(to_float01(X)).permute(0, 3, 1, 2).contiguous()
    yt = torch.from_numpy(y)
    bs = 96
    for step in range(steps):
        idx = torch.randint(0, xt.shape[0], (bs,))
        loss = F.binary_cross_entropy_with_logits(model(xt[idx].to(device)), yt[idx].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log and step % 200 == 0:
            log(f"  detector step {step}: loss {loss.item():.4f}")
    model.eval()

    report = dict(meta)
    # ---- validation on real val frames (1500 pos / 1500 neg sampled), plus controls
    vrows = opening_labels(sidecar, packed, "val", max_frames=None, seed=seed + 1)
    vp = pick([(e, i) for (e, i, l, _) in vrows if l == 1], 1500)
    vn = pick([(e, i) for (e, i, l, c) in vrows if l == 0 and not c], 1500)
    vc = pick([(e, i) for (e, i, l, c) in vrows if c], 400)
    det = OpeningDetector(model, 0.5, report)
    pp = det.prob(_imgs(vp), device) if vp else np.zeros(0)
    pn = det.prob(_imgs(vn), device) if vn else np.zeros(0)
    pc = det.prob(_imgs(vc), device) if vc else np.zeros(0)
    if len(pp) and len(pn):
        tp, fp = float((pp >= .5).sum()), float((pn >= .5).sum())
        report.update(val_pos_rate=float((pp >= .5).mean()), val_neg_rate=float((pn >= .5).mean()),
                      val_precision=tp / max(1.0, tp + fp), val_recall=float((pp >= .5).mean()))
    if len(pc):
        Xc_val = _imgs(vc)
        dark = _dark_mask(Xc_val, thresh=float(np.percentile(
            luminance(to_float01(Xc_val)).mean(axis=(1, 2)), 25)))
        report["no_door_controls"] = int(len(pc))
        report["no_door_control_rate"] = float((pc >= .5).mean())
        report["dark_control_n"] = int(dark.sum())
        if dark.sum():
            report["dark_control_rate"] = float((pc[dark] >= .5).mean())
        # base rates on the whole val split, for the report
        allrows = opening_labels(sidecar, packed, "val", max_frames=None)
        labs = np.array([r[2] for r in allrows])
        report["val_positive_share"] = float(labs.mean())
        report["val_control_share"] = float(np.mean([r[3] for r in allrows]))
        # ---- calibrate the decision threshold on REAL frames so that its positive
        # share equals the sidecar's own base rate.  The net is only ~0.6 precise at
        # the frame level, so a fixed 0.5 cut would bias the real reference; anchoring
        # the threshold to real frames means the generated share is measured against
        # the same operating point.
        cal = pick([(e, i) for (e, i, l, _) in vrows], 4000)
        if cal:
            pc_cal = det.prob(_imgs(cal), device)
            t = float(np.quantile(pc_cal, 1.0 - float(labs.mean())))
            report["calibrated_thresh"] = t
            report["calibrated_pos_rate_real"] = float((pc_cal >= t).mean())
    if out_path:
        det = OpeningDetector(model, report.get("calibrated_thresh", 0.5), report)
        det.save(out_path)
    return det, report


def _dark_mask(frames_u8, thresh=0.10) -> np.ndarray:
    return luminance(to_float01(frames_u8)).mean(axis=(1, 2)) < thresh


def opening_events(pos: np.ndarray) -> dict:
    """Positive frame share plus 'opening events' per 100 frames.

    An event is a rising edge of the detector output (an opening becomes visible)
    that was preceded by at least 4 negative frames (so noise on a positive frame
    does not inflate the count).
    """
    pos = np.asarray(pos).astype(bool)
    events = 0
    gap = 0
    for v in pos:
        if v and gap >= 4:
            events += 1
        gap = 0 if v else gap + 1
    return {
        "positive_share": float(pos.mean()) if len(pos) else 0.0,
        "events_per_100": float(events / max(1, len(pos)) * 100.0),
        "n_frames": int(len(pos)),
        "n_events": int(events),
    }


# --------------------------------------------------------------------------- #
# C. real-versus-generated classifier (two-sample test)
# --------------------------------------------------------------------------- #

def _stack_pool(pool):
    return np.concatenate(pool, axis=0)


def train_realfake(gen_tr, real_tr, gen_te, real_te, steps=900, device="cuda", seed=0, log=None):
    """Held-out AUC of a fresh small CNN telling real from generated frames.

    0.5 = indistinguishable (ideal), 1.0 = trivially separable.  Train and test
    use disjoint clips and disjoint sampling seeds.
    """
    torch.manual_seed(seed)
    X = np.concatenate([gen_tr, real_tr])
    y = np.concatenate([np.ones(len(gen_tr)), np.zeros(len(real_tr))]).astype(np.float32)
    Xt = torch.from_numpy(to_float01(X)).permute(0, 3, 1, 2).contiguous()
    yt = torch.from_numpy(y)

    model = _SmallCNN(width=12).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for step in range(steps):
        idx = torch.randint(0, Xt.shape[0], (64,))
        loss = F.binary_cross_entropy_with_logits(model(Xt[idx].to(device)), yt[idx].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()

    @torch.no_grad()
    def _score(A):
        A = torch.from_numpy(to_float01(np.asarray(A))).permute(0, 3, 1, 2).contiguous()
        out = []
        for i in range(0, A.shape[0], 256):
            out.append(model(A[i: i + 256].to(device)).float().cpu())
        return torch.cat(out).numpy() if out else np.zeros(0)

    sg, sr = _score(gen_te), _score(real_te)
    return {"auc": float(_auc(sg, sr)), "n_train": int(len(X)), "n_test": int(len(sg) + len(sr))}


def _auc(scores_pos, scores_neg) -> float:
    """Rank AUC of P(generated) for generated vs real, with ties at 0.5."""
    a = np.asarray(scores_pos, dtype=np.float64)
    b = np.asarray(scores_neg, dtype=np.float64)
    if not len(a) or not len(b):
        return 0.5
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    ra = ranks[: len(a)].sum()
    return float((ra - len(a) * (len(a) + 1) / 2.0) / (len(a) * len(b)))


# --------------------------------------------------------------------------- #
# H. long-run health
# --------------------------------------------------------------------------- #

def health_stats(seq_u8: np.ndarray, real_ref: np.ndarray | None = None) -> dict:
    """Colour / luminance / dark-region / motion statistics of a frame sequence.

    ``seq_u8`` is (T,H,W,3) uint8 and includes the real context frame at index 0,
    so frame-to-frame statistics are over the generated steps.
    """
    f = to_float01(seq_u8)
    lum = luminance(f)
    dark = lum < 0.06
    out = {
        "mean_rgb": [float(x) for x in f.reshape(-1, 3).mean(axis=0)],
        "mean_luminance": float(lum.mean()),
        "dark_share": float(dark.mean()),
        "near_black_share": float((lum < 0.02).mean()),
        "near_white_share": float((lum > 0.98).mean()),
        "laplacian_var": float(np.var(_laplacian(f))),
    }
    if dark.any():
        px = f[dark]
        out["dark_rgb"] = [float(x) for x in px.mean(axis=0)]
        out["dark_r_minus_b"] = float(px[:, 0].mean() - px[:, 2].mean())
        out["dark_pixels"] = int(dark.sum())
    else:
        out["dark_rgb"] = None
        out["dark_r_minus_b"] = None
        out["dark_pixels"] = 0

    # drift: linear fit of per-frame mean luminance over the sequence
    t = np.arange(len(lum), dtype=np.float64)
    out["luminance_drift_per_frame"] = float(np.polyfit(t, lum.mean(axis=(1, 2)), 1)[0])
    # change and motion
    d = np.abs(np.diff(f, axis=0))
    out["frame_change"] = float(d.mean())
    out["frozen"] = bool(out["frame_change"] < 1e-4)
    out["collapse_black"] = bool(lum.mean() < 0.004 and out["dark_share"] > 0.999)
    out["collapse_white"] = bool(lum.mean() > 0.996)
    out["std"] = float(f.std())
    if real_ref is not None and len(real_ref) > 1:
        rr = to_float01(real_ref)
        out["frame_change_real"] = float(np.abs(np.diff(rr, axis=0)).mean())
        out["laplacian_var_real"] = float(np.var(_laplacian(rr)))
    return out


def motion_px_per_step(seq_u8: np.ndarray) -> float:
    """Mean absolute horizontal content motion per step, in px (all steps).

    Uses the same per-step phase correlation as the Stage 5 gates, averaged over
    the whole sequence.  For long sequences the accumulated value aliases, so this
    is deliberately per-step (small shifts stay inside the estimator's range) and
    is reported as a magnitude.
    """
    from gates import estimate_shift_x

    if len(seq_u8) < 2:
        return 0.0
    vals = [abs(estimate_shift_x(seq_u8[k], seq_u8[k + 1])) for k in range(len(seq_u8) - 1)]
    return float(np.mean(vals))
