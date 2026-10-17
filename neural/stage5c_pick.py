"""Stage 5c — Phase 3: choose which snapshot to present (4.3 item 1).

    .venv/bin/python stage5c_pick.py --run flow96e --out eval_out/stage5c_pick.json

Scores **every** saved snapshot (and the final EMA) with the 4.0d live-path health
numbers on the same 3 val seed contexts, no clicks, 300 steps. The selection rule is the
brief's, with the vision judge dropped (see §4.0c): **the latest snapshot that passes the
band**, and if none passes, the healthiest, ranked by contexts-in-band and then by
distance to the real reference's time statistics. ``val_sample_l1`` is deliberately *not*
used — it is recorded in the table only to show what it would have picked.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

import stage5c_rollout as R

REAL_PACKED = "data/bulk_v1_96x72"


def _step_of(path: str) -> int:
    """The true training step of a checkpoint (from its payload when not a snapshot)."""
    base = os.path.basename(path)
    if base.startswith("snap_"):
        return int(base[5:12])
    try:
        import torch
        return int(torch.load(path, map_location="cpu", weights_only=False).get("step", -1))
    except Exception:                                          # pragma: no cover
        return -1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="flow96e")
    ap.add_argument("--out", default="eval_out/stage5c_pick.json")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--contexts", type=int, default=3)
    ap.add_argument("--min-in-band", type=float, default=2 / 3)
    ap.add_argument("--rule", choices=("healthiest", "latest_healthy"),
                    default="healthiest",
                    help="healthiest: all contexts in band, then closest to real "
                         "(the brief's 'select by health, not by the last step'); "
                         "latest_healthy: the last snapshot above --min-in-band")
    ap.add_argument("--extras", default="", help="extra checkpoints to include")
    args = ap.parse_args(argv)

    real = R.real_reference(REAL_PACKED, n_clips=8, steps=args.steps)["stats"]
    seeds = R.pick_seeds(args.contexts, "val", offset=0)
    run_dir = os.path.join("runs", args.run)
    cands = sorted(glob.glob(os.path.join(run_dir, "snapshots", "snap_*.pt")),
                   key=_step_of)
    for extra in ("best.pt", "last.pt", "final.pt"):
        p = os.path.join(run_dir, extra)
        if os.path.exists(p):
            cands.append(p)
    cands += [x for x in args.extras.split(",") if x]

    real_target = {k: float(real[k]) for k in
                   ("frame_change", "hf_ratio", "luma_mean", "contrast")}
    rows = []
    for path in cands:
        player = R.open_player(path)
        hs = []
        for sd in seeds:
            fr = R.rollout_live(player, sd, steps=args.steps, noise_seed=0)
            h = R.health(fr, real_hf=real["hf_energy"])
            h["seed"] = sd.label
            h["in_band"] = R.in_band(h)["all"]
            hs.append(h)
        mean = {k: float(np.mean([h[k] for h in hs])) for k in
                ("frame_change", "hf_ratio", "contrast", "luma_mean", "near_black")}
        n_in = int(sum(h["in_band"] for h in hs))
        dist = float(np.mean([abs(np.log(max(1e-6, mean[k]) / real_target[k]))
                              for k in real_target]))
        rows.append({"path": path, "step": _step_of(path), "n_in_band": n_in,
                     "n_contexts": len(hs), "health": mean, "log_dist_to_real": dist,
                     "contexts": hs})
        print(f"{os.path.basename(path):22s} step {_step_of(path):>7} "
              f"in-band {n_in}/{len(hs)}  "
              f"fc {mean['frame_change']:.2f} hf {mean['hf_ratio']:.2f} "
              f"ct {mean['contrast']:.1f} lu {mean['luma_mean']:.1f} "
              f"dist {dist:.3f}", flush=True)

    ok = [r for r in rows if r["n_in_band"] >= args.min_in_band * r["n_contexts"]]
    if ok and args.rule == "healthiest":
        best_n = max(r["n_in_band"] for r in ok)
        full = [r for r in ok if r["n_in_band"] == best_n]
        pick = min(full, key=lambda r: r["log_dist_to_real"])
        reason = (f"healthiest ({best_n}/{full[0]['n_contexts']} contexts in band) and "
                  f"closest to the real reference's time statistics "
                  f"(log-dist {pick['log_dist_to_real']:.3f}); "
                  f"the brief's \"select by health, not by the last step\"")
    elif ok:
        pick, reason = max(ok, key=lambda r: r["step"]), "latest snapshot passing the band"
    elif rows:
        pick = max(rows, key=lambda r: (r["n_in_band"], -r["log_dist_to_real"]))
        reason = "no snapshot passes the band: healthiest, then closest to real"
    else:
        raise SystemExit("no candidates found")
    print(f"\nPICK {pick['path']} (step {pick['step']}, in-band "
          f"{pick['n_in_band']}/{pick['n_contexts']}) -- {reason}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"real": real, "seed_contexts": [s.label for s in seeds],
                   "candidates": rows, "pick": pick["path"],
                   "pick_step": pick["step"], "pick_reason": reason,
                   "min_in_band": args.min_in_band}, fh, indent=1)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
