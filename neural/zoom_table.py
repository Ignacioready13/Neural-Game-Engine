"""Stage 5b — the per-step forward-motion (zoom) supervision table.

Why
---
The structured motion path in Stage 5 is translation-only, but walking forward makes
the walls and doorways **expand away from the vanishing point** every frame.  The
generator is therefore asked to invent that expansion from scratch, frame after
frame, which is a large part of why long rollouts smear.

This module measures the expansion from the **sidecar**, not from the pixels: a
doorway at a fixed bearing off-axis has ``screenX = x / z`` for a pinhole camera, so
when the camera moves forward (``z`` shrinks) its bearing grows proportionally.
For two consecutive frames we match doorway candidates by ``doorwayId`` and fit

    screenX_{t+1} = a * screenX_t + b        (a = 1 / scale, b = a residual pan)

by least squares; ``log_zoom = -log(a)`` is positive when the world expands (walking
forward).  With one matched door ``b`` is not identifiable and is dropped; with two
or more the fit separates expansion from a pan, which matters because a turn also
moves every bearing the same way.

This is a *training-only* signal (N1/N2): nothing in the inference path reads it.
"""

from __future__ import annotations

import os

import numpy as np


def _cand_map(cand):
    """{id: (screenX, screenDistance)} for one frame's candidate list."""
    if not cand:
        return {}
    return {str(c["doorwayId"]): (float(c["screenX"]), float(c["screenDistance"]))
            for c in cand}


def sidecar_zoom_table(packed, sidecar, min_abs_x: float = 0.15, log=print):
    """Per-frame log-zoom (frame t -> t+1) for every frame of the packed set.

    ``min_abs_x`` drops matches too close to the screen centre: at ``screenX ~ 0`` the
    bearing ratio is ill-conditioned (a doorway dead ahead barely moves), so those
    pairs carry no information about expansion.
    """
    n = packed.num_frames
    log_zoom = np.zeros(n, dtype=np.float32)
    n_match = np.zeros(n, dtype=np.int16)
    n_doors = np.zeros(n, dtype=np.int16)
    cands = sidecar.candidates
    for t in range(n - 1):
        a, b = _cand_map(cands[t]), _cand_map(cands[t + 1])
        if not a or not b:
            continue
        xs, ys = [], []
        for k, (x1, _) in a.items():
            if k not in b:
                continue
            x2 = b[k][0]
            if abs(x1) < min_abs_x:
                continue
            xs.append(x1)
            ys.append(x2)
        n_match[t] = len(xs)
        n_doors[t] = len(a)
        if not xs:
            continue
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        if len(x) >= 2 and np.ptp(x) > 1e-3:
            A = np.stack([x, np.ones_like(x)], axis=1)
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
            a_fit = float(sol[0])
        else:
            a_fit = float(y[0] / x[0]) if abs(x[0]) > 1e-6 else 1.0
        if not np.isfinite(a_fit) or a_fit <= 1e-3 or a_fit > 5.0:
            continue
        log_zoom[t] = -np.log(a_fit)
    log(f"zoom table: {n} frames, {int((n_match > 0).sum())} with a usable match "
        f"({(n_match > 0).mean():.1%}), {int((n_match >= 2).sum())} with two or more")
    return {"log_zoom": log_zoom, "n_match": n_match, "n_doors": n_doors}


def load_or_compute_zoom(packed, sidecar, packed_path=None, cache_dir=None, log=print):
    packed_path = packed_path or packed.path
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "data", "_cache")
    key = os.path.basename(os.path.normpath(packed_path))
    path = os.path.join(cache_dir, f"{key}_zoom_table.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    out = sidecar_zoom_table(packed, sidecar, log=log)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(path, **out)
    log(f"wrote {path}")
    return out


def cumulative_zoom(table, frames, clip: int = 12.0):
    """Sum of log-zoom over ``frames`` consecutive steps before each index."""
    lz = np.asarray(table["log_zoom"], dtype=np.float64)
    c = np.concatenate([[0.0], np.cumsum(lz)])
    # a cumulative sum for prefix lookups: cumulative_zoom_at[i] = sum(lz[i-clip:i])
    return c


def zoom_prefix(table):
    lz = np.asarray(table["log_zoom"], dtype=np.float64)
    return np.concatenate([[0.0], np.cumsum(lz)])


def validate_against_images(table, packed, n_pairs=200, seed=0, log=print):
    """Cross-check the sidecar zoom against an image-based (optional) estimate.

    The image-based estimate is the brute-force similarity fit in
    ``diagnose_stage5b``; running it on a sample gives an independent check that the
    sidecar bearings really do encode forward motion with the expected sign.
    """
    from diagnose_stage5b import _zoom_shift_fit

    rng = np.random.RandomState(seed)
    lz = table["log_zoom"]
    cand = np.where(np.abs(lz) > 1e-4)[0]
    cand = cand[cand + 1 < packed.num_frames]
    if len(cand) == 0:
        return {"n": 0}
    pick = rng.choice(cand, size=min(n_pairs, len(cand)), replace=False)
    img, sc = [], []
    for i in pick:
        a, b = np.asarray(packed.images[i]), np.asarray(packed.images[i + 1])
        z, _, _ = _zoom_shift_fit(a, b)
        img.append(z)
        sc.append(float(lz[i]))
    img = np.asarray(img)
    sc = np.asarray(sc)
    both = (img > 0) | (sc > 0)
    out = {
        "n": int(len(img)),
        "sidecar_mean": float(sc.mean()),
        "image_mean": float(img.mean()),
        "corr": float(np.corrcoef(sc, img)[0, 1]) if len(img) > 2 else None,
        "sign_agreement_positive": float((both[img > 0]).mean()) if (img > 0).any() else None,
        "sign_agreement": float(((img > 0) == (sc > 0)).mean()),
    }
    log(f"zoom cross-check: {out}")
    return out
