"""Stage 5c — the one steering sanity check (4.3 item 3).

Not the whole scorecard (Stage 5c forbids rebuilding it): the existing, focused
``scorecard.commanded_shift`` instrument, which asks one question — does the frozen
readout still command a sensible image shift once the context is made of the model's
own frames? If the closed loop freezes (ratio → 0) no doorway can ever come into view.

    .venv/bin/python stage5c_steer.py --models flow96,flow96e --out eval_out/stage5c_steer.json
"""

from __future__ import annotations

import argparse
import json
import os

from dataset import PackedDataset
import scorecard as SC
import stage5c_rollout as R


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="flow96")
    ap.add_argument("--packed", default="data/bulk_v1_96x72")
    ap.add_argument("--n-clips", type=int, default=12)
    ap.add_argument("--n-steps", type=int, default=12)
    ap.add_argument("--out", default="eval_out/stage5c_steer.json")
    args = ap.parse_args(argv)

    packed = PackedDataset(args.packed, in_memory=False)
    eps = [e for e in packed.episodes if e["split"] == "val" and e["length"] > 200]
    jobs = [(e["episode_index"], 60) for e in eps[:args.n_clips]]
    res = {}
    for m in [x for x in args.models.split(",") if x]:
        player = R.open_player(m)
        dev = player.device
        r = SC.commanded_shift(packed, player.model, jobs, player.K, 40, 0, dev,
                               n_clips=args.n_clips, n_steps=args.n_steps)
        res[m] = r
        print(f"{m:12s} {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})}",
              flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
