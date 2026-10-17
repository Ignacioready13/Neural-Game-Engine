"""Stage 5c — the optional context-guidance test (4.3 item 2).

Classifier-free guidance on the frame conditioning only (GameNGen Sec 3.3). Inference
only; **no default is changed anywhere**. Reports the 4.0d health numbers and the
per-frame time at guidance 1.0 and at the brief's suggested ~1.5.

    .venv/bin/python stage5c_guidance.py --model runs/flow96e/final.pt --weights 1.5
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import stage5c_rollout as R

REAL_PACKED = "data/bulk_v1_96x72"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", default="1.0,1.5")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--contexts", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="eval_out/stage5c_guidance.json")
    args = ap.parse_args(argv)

    real = R.real_reference(REAL_PACKED, n_clips=8, steps=args.steps)["stats"]
    seeds = R.pick_seeds(args.contexts, "val", offset=0)
    player = R.open_player(args.model)
    model = player.model
    if getattr(model, "null_embed", None) is None:
        raise SystemExit(f"{args.model} has no null conditioning (ctx_dropout=0); "
                         f"guidance is not defined for it")
    res = {"model": args.model, "weights": {}, "real": real}
    for w in [float(x) for x in args.weights.split(",") if x]:
        hs, times = [], []
        for sd in seeds:
            d = R.Drive(model, sd, device=player.device, K=player.K,
                        hist_extra=player.hist_extra)
            for _ in range(args.warmup):
                d.step(0)
            d = R.Drive(model, sd, device=player.device, K=player.K,
                        hist_extra=player.hist_extra)
            t0 = time.perf_counter()
            fr = d.generate(args.steps, noise_seed=0, guidance=w)
            times.append((time.perf_counter() - t0) * 1000.0 / args.steps)
            h = R.health(fr, real_hf=real["hf_energy"])
            h["in_band"] = R.in_band(h)["all"]
            hs.append(h)
        mean = {k: float(np.mean([h[k] for h in hs])) for k in
                ("frame_change", "hf_ratio", "contrast", "luma_mean", "near_black")}
        entry = {"health": mean, "n_in_band": int(sum(h["in_band"] for h in hs)),
                 "n_contexts": len(hs), "ms_per_frame": float(np.mean(times))}
        res["weights"][str(w)] = entry
        print(f"guidance {w:>4}: in-band {entry['n_in_band']}/{entry['n_contexts']}  "
              f"fc {mean['frame_change']:.2f} hf {mean['hf_ratio']:.2f} "
              f"ct {mean['contrast']:.1f} lu {mean['luma_mean']:.1f}  "
              f"{entry['ms_per_frame']:.2f} ms/frame", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
