"""Per-frame per-step horizontal image-shift table (Stage 5, additive).

The steering core (``steer.py``) needs a *supervision target* for the learned
ego-motion readout: how far the view actually moves, in pixels, over a step. The
right quantity is the one the gates measure — the phase-correlation horizontal
shift between consecutive frames (``metrics.estimate_shift_x``) — not raw heading.
This module precomputes that table once per packed set and caches it as a ``.npy``
so training never pays for an FFT per window.

The table is vectorized per episode (``np.fft`` batches over the leading axis), so
a whole 876k-frame set costs a couple of minutes rather than hours.

    shift[i]  = dx the *content* moves between frame i and frame i+1
              = ``estimate_shift_x(frame_i, frame_i+1)``
              = 0 on an episode's final frame (no transition).

Cumulative shift over m steps = ``np.cumsum`` of the table within an episode.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from dataset import PackedDataset
from metrics import _subpixel_peak


def _window_2d(h: int, w: int) -> np.ndarray:
    return np.outer(np.hanning(h), np.hanning(w))


def _episode_shifts(gray: np.ndarray, win: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive-pair shifts for one episode's grayscale frames.

    ``gray`` is [L, h, w] float64. Returns ``(dx, dy)`` of length ``L-1`` in the
    ``estimate_shift_x`` convention (positive dx = content moved right)."""
    L = gray.shape[0]
    if L < 2:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    x = gray - gray.mean(axis=(1, 2), keepdims=True)
    F = np.fft.rfft2(x * win, axes=(1, 2))
    R = F[:-1] * np.conj(F[1:])
    R /= (np.abs(R) + 1e-12)
    corr = np.fft.irfft2(R, s=gray.shape[1:], axes=(1, 2))
    corr = np.abs(corr)
    flat = corr.reshape(corr.shape[0], -1)
    idx = flat.argmax(axis=1)
    h, w = gray.shape[1:]
    ys, xs = np.divmod(idx, w)
    dx = np.zeros(len(idx), np.float64)
    dy = np.zeros(len(idx), np.float64)
    for k in range(len(idx)):
        ddx, ddy = _subpixel_peak(corr[k], int(ys[k]), int(xs[k]))
        fx = float(xs[k]) + ddx
        fy = float(ys[k]) + ddy
        if fx > w / 2.0:
            fx -= w
        if fy > h / 2.0:
            fy -= h
        dx[k], dy[k] = -fx, -fy
    return dx.astype(np.float32), dy.astype(np.float32)


def compute_shift_table(packed: PackedDataset, progress: bool = True) -> dict:
    """Return ``{"dx": float32[num_frames], "dy": float32[num_frames]}``."""
    n = packed.num_frames
    h, w = packed.height, packed.width
    dx = np.zeros(n, np.float32)
    dy = np.zeros(n, np.float32)
    win = _window_2d(h, w)
    t0 = time.time()
    for i, e in enumerate(packed.episodes):
        off = e["frame_offset"]
        L = e["length"]
        frames = np.asarray(packed.images[off: off + L], dtype=np.float64)
        gray = frames.mean(axis=-1)
        ddx, ddy = _episode_shifts(gray, win)
        dx[off: off + len(ddx)] = ddx
        dy[off: off + len(ddy)] = ddy
        if progress and (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(packed.episodes)} episodes, {time.time() - t0:.0f}s",
                  flush=True)
    if progress:
        print(f"shift table done in {time.time() - t0:.0f}s", flush=True)
    return {"dx": dx, "dy": dy}


def cache_dir(packed_path: str) -> str:
    """Cache directory kept *outside* the packed set (the datasets are read-only)."""
    d = os.path.join(os.path.dirname(os.path.abspath(packed_path)), "_cache")
    os.makedirs(d, exist_ok=True)
    return d


def cache_path(packed_path: str) -> str:
    return os.path.join(cache_dir(packed_path),
                        os.path.basename(os.path.abspath(packed_path)) + "_shift_table.npz")


def load_or_compute(packed: PackedDataset, packed_path: str | None = None,
                    progress: bool = True) -> dict:
    """Load the cached table for `packed`, computing and caching it if absent."""
    p = cache_path(packed_path or packed.path)
    if os.path.exists(p):
        z = np.load(p)
        return {"dx": z["dx"], "dy": z["dy"]}
    t = compute_shift_table(packed, progress=progress)
    np.savez_compressed(p, dx=t["dx"], dy=t["dy"])
    return t


def cumulative_shift(table: dict, episode_record: dict, frame: int, horizons) -> np.ndarray:
    """Cumulative content shift from ``frame`` over each horizon in ``horizons``.

    Clipped at the episode boundary (a horizon that runs past the last frame is
    filled with the last available cumulative value)."""
    off = episode_record["frame_offset"]
    L = episode_record["length"]
    dx = table["dx"]
    out = np.zeros(len(horizons), np.float64)
    for j, m in enumerate(horizons):
        lo = min(frame, L - 1)
        hi = min(frame + m, L - 1)
        out[j] = float(dx[off + lo: off + hi].sum())
    return out


def heading_shift_table(packed, phase_table: dict | None = None,
                        cache: bool = True, progress: bool = True) -> dict:
    """A *denoised* per-step horizontal-shift target derived from the sidecar heading.

    The phase-correlation per-step estimate (``compute_shift_table``) has a noise
    floor of ~0.4 px against a true per-step signal of ~0.5 px, so a regression on
    it attenuates: the shut-down regime matters here, since at 20 Hz one step of a
    turn is a fraction of a pixel. The sidecar's ``playerHeading`` is exact, and
    image shift is proportional to heading change (a camera-rotation model), so

        dshift[t] = scale * (heading[t+1] - heading[t])        (wrapped to ±pi)

    with ``scale`` fitted once (px per rad; typically ≈ −35 px/rad at 64×48 and
    ≈ −53 at 96×72 — negative because a positive heading change is a turn toward
    positive screenX, which moves the content *left*). The fit is reported with its
    correlation so the scaling is auditable. Training-only: this table is never an
    input at inference (N1/N2).
    """
    from analysis import load_analysis
    path = os.path.join(cache_dir(packed.path),
                        os.path.basename(packed.path) + "_heading_shift_table.npz")
    if cache and os.path.exists(path):
        z = np.load(path)
        return {"dx": z["dx"], "scale": float(z["scale"]), "corr": float(z["corr"])}

    a = load_analysis(packed)
    n = packed.num_frames
    dh = np.zeros(n, np.float64)
    for e in packed.episodes:
        off, L = e["frame_offset"], e["length"]
        h = np.asarray(a.heading[off: off + L], dtype=np.float64)
        d = np.diff(h)
        d = (d + np.pi) % (2 * np.pi) - np.pi
        dh[off: off + L - 1] = np.nan_to_num(d, nan=0.0)
    scale, corr = 1.0, float("nan")
    if phase_table is not None:
        dx = phase_table["dx"].astype(np.float64)
        m = (np.abs(dh) > 1e-4) & (np.abs(dx) > 1e-6)
        if m.sum() > 100:
            scale = float(np.polyfit(dh[m], dx[m], 1)[0])
            corr = float(np.corrcoef(dh[m], dx[m])[0, 1])
    out = {"dx": (scale * dh).astype(np.float32), "scale": scale, "corr": corr}
    if progress:
        print(f"heading shift table: scale {scale:.2f} px/rad, corr {corr:.3f}", flush=True)
    if cache:
        np.savez_compressed(path, dx=out["dx"], scale=scale, corr=corr)
    return out


def load_target_table(packed, packed_path: str | None, target: str = "heading") -> dict:
    """Return the supervision table for ``target`` ('heading' or 'phase')."""
    phase = load_or_compute(packed, packed_path)
    if target == "phase":
        return phase
    return heading_shift_table(packed, phase)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Precompute the per-step image-shift table")
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", default=None, help="output .npz (default: data/_cache/<pack>_shift_table.npz)")
    args = ap.parse_args(argv)

    packed = PackedDataset(args.packed)
    table = compute_shift_table(packed)
    out = args.out or cache_path(args.packed)
    np.savez_compressed(out, dx=table["dx"], dy=table["dy"])
    dx, dy = table["dx"], table["dy"]
    print(f"wrote {out}")
    print(f"dx: mean|.| {np.abs(dx).mean():.4f} px  p99|.| {np.percentile(np.abs(dx), 99):.3f} px")
    print(f"dy: mean|.| {np.abs(dy).mean():.4f} px  p99|.| {np.percentile(np.abs(dy), 99):.3f} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
