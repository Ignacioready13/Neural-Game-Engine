"""Stage 5c probe — the one short headless experiment behind §4.0a, §4.0b and §4.0d.

    .venv/bin/python stage5c_probe.py --out eval_out/stage5c_probe.json

It (1) confirms the path audit by showing what a *frozen* noise tensor does versus the
player's fresh-per-frame noise, (2) sweeps the temporal noise correlation rho, and
(3) computes the 4.0d health numbers for flow96 / flow96c / flow96d and for real
frames, so the bands can be checked against the user's known verdicts.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import stage5c_rollout as R

PACKED_96 = "data/bulk_v1_96x72"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_out/stage5c_probe.json")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--models", default="flow96,flow96c,flow96d")
    ap.add_argument("--rhos", default="0,0.9,0.99,1.0")
    ap.add_argument("--clicks", default="30:0.6,80:-0.7,150:0.8,220:-0.4")
    args = ap.parse_args(argv)

    t0 = time.time()
    real = R.real_reference(PACKED_96, n_clips=8, steps=args.steps)
    real_hf = real["stats"]["hf_energy"]
    print(f"real reference: {json.dumps(real['stats'])}", flush=True)

    seeds = R.pick_seeds(3, "val", offset=0)
    print("seed contexts: " + ", ".join(s.label for s in seeds), flush=True)

    models = [m for m in args.models.split(",") if m]
    rhos = [float(r) for r in args.rhos.split(",") if r]
    clicks = [(int(a.split(":")[0]), float(a.split(":")[1]))
              for a in args.clicks.split(",") if a]

    result = {"steps": args.steps, "real": real["stats"], "seeds":
              [s.label for s in seeds], "models": {}}

    # --- 4.0a confirmation + 4.0d health + 4.0b rho sweep ---------------------- #
    for m in models:
        player = R.open_player(m)
        entry = {"frozen_noise": {}, "rho": {}, "real_click_health": {}}

        # frozen noise, no-click: the scorecard's path (rho=1)
        # fresh noise, no-click: the live player's path (rho=0)
        for rho in rhos:
            rows = []
            for sd in seeds:
                fr = R.rollout_correlated(player, sd, steps=args.steps,
                                          noise_seed=0, rho=rho)
                rows.append(R.health(fr, real_hf=real_hf))
            agg = {k: float(np.mean([r[k] for r in rows]))
                   for k in ("frame_change", "luma_mean", "contrast", "near_black",
                             "hf_ratio")}
            if rho == 0.0:
                entry["rho"]["0.0_live"] = agg
            entry["rho"][f"{rho}"] = agg
            print(f"  {m} rho={rho}: {json.dumps(agg)}", flush=True)

        # bit-exactness check: rollout_correlated(rho=0) == rollout_live
        a = R.rollout_live(player, seeds[0], steps=8, noise_seed=3)
        b = R.rollout_correlated(player, seeds[0], steps=8, noise_seed=3, rho=0.0)
        entry["bitexact_r0"] = bool(np.array_equal(a, b))
        print(f"  {m} rho=0 bit-identical to Player.generate: {entry['bitexact_r0']}",
              flush=True)

        # real-click health (the secondary goal): live path
        for sd in seeds:
            fr = R.rollout_live(player, sd, steps=args.steps, clicks=clicks,
                                noise_seed=0)
            entry["real_click_health"][sd.label] = R.health(fr, real_hf=real_hf)
        result["models"][m] = entry

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print(f"wrote {args.out} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
