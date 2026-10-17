"""Stage 5c — the health table (verification item 5), live path only.

    .venv/bin/python stage5c_health.py --models flow96,flow96c,flow96d \
        --seed-contexts 3 --noise-seeds 3 --steps 300 --out eval_out/stage5c_health.json

One row per (model, seed context, noise seed): the 4.0d numbers plus the pass-band
check. No clicks by default (``--clicks`` switches to a real-click scenario).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import stage5c_rollout as R

REAL_PACKED = "data/bulk_v1_96x72"
DEFAULT_CLICKS = "30:0.6,80:-0.7,150:0.8,220:-0.4"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="flow96,flow96c,flow96d")
    ap.add_argument("--seed-contexts", type=int, default=3)
    ap.add_argument("--noise-seeds", default="0")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--clicks", default="")
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--out", default="eval_out/stage5c_health.json")
    ap.add_argument("--real-clips", type=int, default=8)
    args = ap.parse_args(argv)

    real = R.real_reference(REAL_PACKED, n_clips=args.real_clips, steps=args.steps)
    real_hf = real["stats"]["hf_energy"]
    clicks = ([(int(a.split(":")[0]), float(a.split(":")[1]))
               for a in args.clicks.split(",") if a] if args.clicks else None)
    noise_seeds = [int(x) for x in str(args.noise_seeds).split(",") if x]
    seeds = R.pick_seeds(args.seed_contexts, "val", offset=0)

    res = {"real": real["stats"], "steps": args.steps, "rho": args.rho,
           "clicks": args.clicks, "models": {}, "seed_contexts":
           [s.label for s in seeds], "noise_seeds": noise_seeds}

    for m in [x for x in args.models.split(",") if x]:
        player = R.open_player(m)
        rows = []
        for sd in seeds:
            for ns in noise_seeds:
                fr = R.rollout_live(player, sd, steps=args.steps, clicks=clicks,
                                    noise_seed=ns, rho=args.rho)
                h = R.health(fr, real_hf=real_hf)
                h.update(seed=sd.label, noise_seed=ns,
                         in_band=R.in_band(h)["all"])
                rows.append(h)
        agg = {}
        for k in ("frame_change", "hf_ratio", "contrast", "luma_mean", "near_black"):
            agg[k] = float(np.mean([r[k] for r in rows]))
        agg["mean_in_band"] = float(np.mean([r["in_band"] for r in rows]))
        res["models"][m] = {"rows": rows, "mean": agg}
        print(f"{m:10s} mean {json.dumps({k: round(v, 3) for k, v in agg.items()})}",
              flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
