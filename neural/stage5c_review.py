"""Stage 5c — the Phase 3 review artifacts for the user (4.3 item 5).

    .venv/bin/python stage5c_review.py --pick flow96e --out review/stage5c

Everything here comes from the live-path rollouts in ``stage5c_rollout`` (the player's
own maths). Nothing is written twice and nothing existing is touched; every file has
the ``stage5c_`` prefix. **I do not look at any of these.**
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

import stage5c_rollout as R

STEPS = (0, 20, 60, 120, 200, 300)


def _guard(path: str):
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="flow96,flow96c")
    ap.add_argument("--pick", default="flow96e")
    ap.add_argument("--out", default="review/stage5c")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--clicks", default="40:0.7,110:-0.8,190:0.6,260:-0.9")
    ap.add_argument("--gif-frames", type=int, default=300)
    ap.add_argument("--prefer-at", type=int, default=0,
                    help="a chosen snapshot step to strip, if that is not --pick")
    ap.add_argument("--skip-gif", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    models = [m for m in args.models.split(",") if m]
    if args.pick and args.pick not in models:
        models.append(args.pick)
    seeds = R.pick_seeds(args.seeds, "val", offset=0)
    clicks = [(int(a.split(":")[0]), float(a.split(":")[1]))
              for a in args.clicks.split(",") if a]
    made = []

    for m in models:
        player = R.open_player(m)
        import play
        for si, sd in enumerate(seeds):
            fr = R.rollout_live(player, sd, steps=args.steps, noise_seed=0)
            p = os.path.join(args.out, f"stage5c_noclick_{m}_seed{si}.png")
            _guard(p)
            play.write_strip(fr, p, every=args.every, scale=2)
            made.append(p)
        for si, sd in enumerate(seeds):
            fr = R.rollout_live(player, sd, steps=args.steps, clicks=clicks,
                                noise_seed=0)
            p = os.path.join(args.out, f"stage5c_realclick_{m}_seed{si}.png")
            _guard(p)
            play.write_strip(fr, p, every=args.every, scale=2)
            made.append(p)
        print(f"[review] {m}: {2 * len(seeds)} strips", flush=True)

    if not args.skip_gif and args.pick:
        p = os.path.join(args.out, f"stage5c_{args.pick}_noclick.gif")
        _guard(p)
        cmd = [sys.executable, "play.py", "--model", args.pick, "--headless",
               "--frames", str(args.gif_frames), "--out", p, "--gif-scale", "3",
               "--seed-context", f"{int(seeds[0]['episode'])}:{int(seeds[0]['start'])}",
               "--noise-seed", "0"]
        print("[review] " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip(), flush=True)
        if os.path.exists(p):
            made.append(p)
    print(f"[review] {len(made)} artifacts in {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
