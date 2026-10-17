"""Stage 5c — build the 4.0c contact sheets (live-path no-click rollouts + real).

    .venv/bin/python stage5c_sheets.py --out review/stage5c

Each sheet is one row of the frames at steps 0, 20, 60, 120, 200, 300 of a 300-step
no-click live-path rollout (or of real consecutive frames). Nothing is judged here.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import stage5c_rollout as R

STEPS = (0, 20, 60, 120, 200, 300)
PACKED = {"flow96": "data/bulk_v1_96x72", "flow96c": "data/bulk_v1_96x72",
          "flow96d": "data/bulk_v1_96x72"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="review/stage5c")
    ap.add_argument("--models", default="flow96,flow96c,flow96d")
    ap.add_argument("--seeds", default="2")
    ap.add_argument("--rho", type=float, default=0.0)
    ap.add_argument("--prefix", default="stage5c_calib")
    ap.add_argument("--real-clips", type=int, default=2)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    seeds = R.pick_seeds(int(args.seeds), "val", offset=0)
    made = []
    for m in [x for x in args.models.split(",") if x]:
        player = R.open_player(m)
        for si, sd in enumerate(seeds):
            fr = R.rollout_live(player, sd, steps=300, noise_seed=0, rho=args.rho)
            p = os.path.join(args.out, f"{args.prefix}_{m}_seed{si}_{sd.label.replace(' ', '')}.png"
                             .replace("@", "at"))
            R.contact_sheet(fr, p, STEPS, scale=3)
            made.append(p)
            print("wrote", p, flush=True)

    real = R.real_reference(PACKED["flow96"], n_clips=args.real_clips, steps=300)
    for ci in range(real["n_clips"]):
        p = os.path.join(args.out, f"{args.prefix}_real_clip{ci}.png")
        R.contact_sheet(real["frames"][ci], p, STEPS, scale=3)
        made.append(p)
        print("wrote", p, flush=True)
    print(f"{len(made)} sheets in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
