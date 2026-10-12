"""Stage 4B coverage counter: qualifying click-counterfactual contexts per split.

Read-only. Counts, for a packed set, the *qualifying counterfactual contexts*
defined by the Stage 4 click probe — ``probe_click._select_contexts`` (two
reachable doors on opposite sides of screen centre, both within the probe's
``_DOOR_NEAR`` screen-distance threshold, mid-episode) — near vs far, per split
and per 1,000 frames, plus the direction-balance and click-onset statistics the
Stage 4B brief asks for.

This module deliberately reuses ``probe_click._select_contexts`` (do not
reimplement the definition); it only adds the counting/aggregation around it.

Run: .venv/bin/python count_contexts.py --packed data/pilot_v1_64x48
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from analysis import load_analysis
from dataset import PackedDataset
from probe_click import _NEAR_SPLIT, _select_contexts

_SPLITS = ("train", "val", "test")
_HUGE = 10 ** 9  # effectively "no per-episode cap" when counting coverage


def _frames_and_episodes(packed, split):
    eps = packed.episodes_by_split[split]
    return int(sum(e["length"] for e in eps)), len(eps)


def _context_counts(packed, sidecar, split, K, H):
    ctx = _select_contexts(packed, sidecar, K, H, max_per_ep=_HUGE, split=split)
    near = [c for c in ctx if min(c[4], c[5]) < _NEAR_SPLIT]
    far = [c for c in ctx if min(c[4], c[5]) >= _NEAR_SPLIT]
    return ctx, near, far


def _heading_balance(sidecar, packed, split, turn_deg=0.5):
    """Fraction of signed heading changes that are positive (and negative).

    Reported over all steps and restricted to *turn* windows (|d| >= turn_deg),
    which is the split Stage 4 reported as 42.5% / 57.5%."""
    pos = neg = zero = 0
    tpos = tneg = 0
    for e in packed.episodes_by_split[split]:
        off, L = e["frame_offset"], e["length"]
        seg = sidecar.heading[off:off + L]
        d = (seg[1:] - seg[:-1] + np.pi) % (2 * np.pi) - np.pi
        d = d[~np.isnan(d)]
        pos += int((d > 0).sum())
        neg += int((d < 0).sum())
        zero += int((d == 0).sum())
        turn = np.abs(d) >= np.deg2rad(turn_deg)
        tpos += int((d[turn] > 0).sum())
        tneg += int((d[turn] < 0).sum())
    n = pos + neg + zero
    tn = tpos + tneg
    return {"positive": pos, "negative": neg, "zero": zero, "n": n,
            "positive_share": pos / n if n else float("nan"),
            "negative_share": neg / n if n else float("nan"),
            "turn_positive": tpos, "turn_negative": tneg, "turn_n": tn,
            "turn_positive_share": tpos / tn if tn else float("nan"),
            "turn_negative_share": tneg / tn if tn else float("nan")}


def _reachable_door_balance(sidecar, packed, split):
    """Left/right balance of reachable candidates' screenX."""
    left = right = 0
    for e in packed.episodes_by_split[split]:
        off, L = e["frame_offset"], e["length"]
        for g in range(off, off + L):
            for c in (sidecar.candidates[g] or ()):
                if not c["reachable"]:
                    continue
                if c["screenX"] < 0:
                    left += 1
                elif c["screenX"] > 0:
                    right += 1
    n = left + right
    return {"left": left, "right": right, "n": n,
            "left_share": left / n if n else float("nan"),
            "right_share": right / n if n else float("nan")}


def _clicks(sidecar, packed, split):
    """Accepted clicks, click-resolved turn direction, onsets and retargets."""
    pos = neg = zero = 0
    accepted = onsets = retargets = 0
    for e in packed.episodes_by_split[split]:
        off, L = e["frame_offset"], e["length"]
        cac = sidecar.clickAccepted[off:off + L]
        for g in np.where(cac == 1.0)[0].tolist():
            if g >= L - 1:
                continue
            accepted += 1
            d = sidecar.heading_change(e["episode_index"], g)
            if not np.isnan(d):
                if d > 0:
                    pos += 1
                elif d < 0:
                    neg += 1
                else:
                    zero += 1
            prev = sidecar.targetDoor[off + g]
            nxt = sidecar.targetDoor[off + g + 1]
            if nxt != prev:
                onsets += 1
                if prev is not None:
                    retargets += 1
    n = pos + neg + zero
    return {"accepted": accepted, "onsets": onsets, "retargets": retargets,
            "turn_positive": pos, "turn_negative": neg, "turn_zero": zero,
            "turn_positive_share": pos / n if n else float("nan"),
            "turn_negative_share": neg / n if n else float("nan")}


def report(packed, K, H, splits=_SPLITS):
    sidecar = load_analysis(packed)
    out = {"packed": packed.path, "K": K, "H": H, "splits": {}}
    for split in splits:
        frames, episodes = _frames_and_episodes(packed, split)
        ctx, near, far = _context_counts(packed, sidecar, split, K, H)
        per1000 = 1000.0 * len(ctx) / frames if frames else float("nan")
        near1000 = 1000.0 * len(near) / frames if frames else float("nan")
        stats = {
            "frames": frames,
            "episodes": episodes,
            "qualifying": len(ctx),
            "qualifying_per1000": per1000,
            "qualifying_episodes": len({c[0] for c in ctx}),
            "near": len(near),
            "near_per1000": near1000,
            "near_episodes": len({c[0] for c in near}),
            "far": len(far),
            "far_episodes": len({c[0] for c in far}),
            "heading": _heading_balance(sidecar, packed, split),
            "reachable_doors": _reachable_door_balance(sidecar, packed, split),
            "clicks": _clicks(sidecar, packed, split),
        }
        out["splits"][split] = stats
    return out


def _print(res):
    print(f"packed: {res['packed']}  (K={res['K']}, H={res['H']})")
    for split, s in res["splits"].items():
        print(f"\n== {split}: {s['episodes']} episodes, {s['frames']} frames ==")
        print(f"  qualifying contexts: {s['qualifying']} "
              f"({s['qualifying_per1000']:.2f}/1000 frames, "
              f"{s['qualifying_episodes']} episodes)")
        print(f"    near: {s['near']} ({s['near_per1000']:.2f}/1000, {s['near_episodes']} eps)"
              f"   far: {s['far']} ({s['far_episodes']} eps)")
        h = s["heading"]
        print(f"  signed heading change: +{h['positive_share']*100:.1f}% / "
              f"-{h['negative_share']*100:.1f}%  (n={h['n']}); "
              f"turn windows (|d|>=0.5deg): +{h['turn_positive_share']*100:.1f}% / "
              f"-{h['turn_negative_share']*100:.1f}%  (n={h['turn_n']})")
        rd = s["reachable_doors"]
        print(f"  reachable doors screenX: left {rd['left_share']*100:.1f}% / "
              f"right {rd['right_share']*100:.1f}%  (n={rd['n']})")
        c = s["clicks"]
        print(f"  accepted clicks: {c['accepted']}  onsets: {c['onsets']}  "
              f"retargets: {c['retargets']}")
        print(f"  click-resolved turn dir: +{c['turn_positive_share']*100:.1f}% / "
              f"-{c['turn_negative_share']*100:.1f}%  (n={c['turn_positive']+c['turn_negative']+c['turn_zero']})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--splits", nargs="+", default=list(_SPLITS))
    ap.add_argument("--json", help="write the full result JSON here")
    args = ap.parse_args(argv)

    packed = PackedDataset(args.packed)  # memmap: sidecar-only work, no image reads
    res = report(packed, args.K, args.H, tuple(args.splits))
    _print(res)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
