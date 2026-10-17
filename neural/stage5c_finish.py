"""Stage 5c — Phase 3: pick, register, guidance, steering, speed, review artifacts.

Run it once the main run has finished (the driver registers `flow96e_final` itself):

    .venv/bin/python stage5c_finish.py --run flow96e --pick-name flow96e

Order and time budget follow the brief (§4.3): the pick and the registration first
because they are the deliverables, then the optional extras, then the artifacts. Every
step is skipped rather than allowed to overrun if it fails; nothing existing is written.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import stage5c_pick
import stage5c_guidance
import stage5c_steer
import stage5c_review


def _run(label, fn, *a, **k):
    t0 = time.time()
    print(f"\n=== {label} ===", flush=True)
    try:
        rc = fn(*a, **k)
        print(f"--- {label}: rc={rc} in {time.time() - t0:.0f}s", flush=True)
        return rc
    except SystemExit as e:
        print(f"--- {label}: SystemExit {e} in {time.time() - t0:.0f}s", flush=True)
        return e.code
    except Exception as e:                                     # pragma: no cover
        print(f"--- {label}: FAILED {type(e).__name__}: {e}", flush=True)
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="flow96e")
    ap.add_argument("--pick-name", default="flow96e")
    ap.add_argument("--skip", default="", help="comma list of extras to skip")
    ap.add_argument("--pick-json", default="eval_out/stage5c_pick.json")
    args = ap.parse_args(argv)
    skip = {s for s in args.skip.split(",") if s}
    summary = {}

    _run("pick", stage5c_pick.main, ["--run", args.run, "--out", args.pick_json])
    with open(args.pick_json) as fh:
        pick = json.load(fh)
    picked = pick["pick"]
    summary["pick"] = {"path": picked, "step": pick["pick_step"],
                       "reason": pick["pick_reason"]}
    print(f"\nPICK = {picked} (step {pick['pick_step']}): {pick['pick_reason']}",
          flush=True)

    # --- register the pick (never touching an existing entry) ------------------ #
    def register_pick():
        cmd = [sys.executable, "register_model.py", "--name", args.pick_name,
               "--generator", picked, "--core", "runs/steer96/best.pt",
               "--parent", "flow96c", "--not-selected",
               "--description",
               f"{args.pick_name}: Stage 5c coherence pick -- "
               f"{os.path.basename(picked)} (step {pick['pick_step']}) of run "
               f"{args.run}, chosen as the latest snapshot inside the 4.0d health band "
               f"(vision judge dropped, see STAGE5C-REPORT.md 4.0c)",
               "--notes", json.dumps({"health": next(
                   (c["health"] for c in pick["candidates"] if c["path"] == picked), {}),
                   "in_band": next((c["n_in_band"] for c in pick["candidates"]
                                    if c["path"] == picked), None),
                   "real": pick["real"]}),
               "--packed", "data/bulk_v1_96x72"]
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip(), flush=True)
        summary["register"] = {"rc": r.returncode,
                               "out": (r.stdout or r.stderr).strip()}
        return r.returncode
    _run("register the pick", register_pick)

    # --- extras --------------------------------------------------------------- #
    if "guidance" not in skip:
        _run("guidance", stage5c_guidance.main,
             ["--model", picked, "--weights", "1.0,1.5",
              "--out", "eval_out/stage5c_guidance.json"])
    if "steer" not in skip:
        _run("steering quick-check", stage5c_steer.main,
             ["--models", f"flow96,{picked}",
              "--out", "eval_out/stage5c_steer.json"])
    if "speed" not in skip:
        def speed():
            cmd = [sys.executable, "play.py", "--model", picked, "--headless",
                   "--frames", "500", "--pace",
                   "--report-frames", "eval_out/stage5c_frames_pick.json"]
            print(" ".join(cmd), flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True)
            print(r.stdout.strip() or r.stderr.strip(), flush=True)
            return r.returncode
        _run("speed (paced 500)", speed)

    # --- artifacts ------------------------------------------------------------ #
    if "review" not in skip:
        _run("review artifacts", stage5c_review.main,
             ["--pick", args.pick_name, "--out", "review/stage5c"])

    with open("eval_out/stage5c_finish.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("\nPHASE 3 SUMMARY\n" + json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
