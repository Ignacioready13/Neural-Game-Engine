"""Stage 4B per-persona statistics: does each persona earn its weight?

For a packed set (+ its engine source dir for `aborted.jsonl`) it reports, per
split and overall: each persona's frame share, its qualifying-counterfactual-
context yield (contexts/1,000 persona-frames, reusing the probe selector), and
its abort exposure (aborts attributed to the persona active at the crash, per
1,000 persona-frames) — so dither/spammer can be checked against both the yield
they buy and the aborts they cost.

Read-only. Run:
  .venv/bin/python persona_stats.py --packed data/bulk_v1_96x72 \
      --source ../engine/data/bulk_v1 --json /tmp/bulk_persona.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np

from analysis import load_analysis
from dataset import PackedDataset
from probe_click import _select_contexts

_SPLITS = ("train", "val", "test")
_K, _H = 4, 1


def _aborts(source):
    out = []
    path = os.path.join(source, "aborted.jsonl")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    return out


def analyze(packed, source, splits=_SPLITS, K=_K, H=_H):
    sidecar = load_analysis(packed)
    personas = ("idle", "decisive", "late", "dither", "prober", "spammer", "uniform")
    res = {"packed": packed.path, "source": os.path.abspath(source), "splits": {}}
    for split in splits:
        frames = collections.Counter()
        for e in packed.episodes_by_split[split]:
            off, L = e["frame_offset"], e["length"]
            for pe in sidecar.persona[off:off + L]:
                frames[pe] += 1
        n_frames = sum(frames.values())
        ctx = _select_contexts(packed, sidecar, K, H, max_per_ep=10 ** 9, split=split)
        qual = collections.Counter()
        for (e, s, *_rest) in ctx:
            qual[sidecar.persona[packed.episodes[e]["frame_offset"] + s + K - 1]] += 1
        res["splits"][split] = {
            "frames": n_frames,
            "episodes": len(packed.episodes_by_split[split]),
            "qualifying": len(ctx),
            "per_persona": {
                pe: {
                    "frames": frames.get(pe, 0),
                    "frame_share": frames.get(pe, 0) / n_frames if n_frames else 0.0,
                    "qualifying": qual.get(pe, 0),
                    "yield_per1000": (1000.0 * qual.get(pe, 0) / frames[pe]) if frames.get(pe) else 0.0,
                }
                for pe in personas
            },
        }
    # aborts, attributed to the persona active at the crash
    aborts = _aborts(source)
    aborts_p = collections.Counter(a.get("persona") for a in aborts)
    res["aborts"] = {
        "total": len(aborts),
        "by_persona": {pe: aborts_p.get(pe, 0) for pe in personas},
        "levels": dict(collections.Counter(a.get("level") for a in aborts)),
        "records": [
            {"episodeId": a.get("episodeId"), "seed": a.get("seed"), "split": a.get("split"),
             "persona": a.get("persona"), "level": a.get("level"),
             "crashFrameIndex": a.get("crashFrameIndex"), "framesKept": a.get("framesKept")}
            for a in aborts
        ],
    }
    # per-persona abort exposure (aborts per 1,000 persona-frames, all splits)
    all_frames = collections.Counter()
    for split in splits:
        for pe, d in res["splits"][split]["per_persona"].items():
            all_frames[pe] += d["frames"]
    res["aborts"]["per1000_persona_frames"] = {
        pe: (1000.0 * aborts_p.get(pe, 0) / all_frames[pe]) if all_frames[pe] else 0.0
        for pe in personas
    }
    res["persona_frames_total"] = dict(all_frames)
    return res


def _print(res):
    print(f"packed: {res['packed']}")
    for split, s in res["splits"].items():
        print(f"\n== {split}: {s['episodes']} eps, {s['frames']} frames, "
              f"{s['qualifying']} qualifying ==")
        print(f"  {'persona':9s} {'share%':>7s} {'qual':>7s} {'yield/1k':>9s}")
        for pe, d in sorted(s["per_persona"].items(), key=lambda kv: -kv[1]["frames"]):
            print(f"  {pe:9s} {d['frame_share']*100:7.1f} {d['qualifying']:7d} {d['yield_per1000']:9.1f}")
    a = res["aborts"]
    print(f"\n== aborts: {a['total']} total, levels {a['levels']} ==")
    print(f"  {'persona':9s} {'aborts':>7s} {'aborts/1k frames':>16s}")
    for pe, n in sorted(a["by_persona"].items(), key=lambda kv: -kv[1]):
        print(f"  {pe:9s} {n:7d} {a['per1000_persona_frames'][pe]:16.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--splits", nargs="+", default=list(_SPLITS))
    ap.add_argument("--json", help="write the full result JSON here")
    args = ap.parse_args(argv)
    packed = PackedDataset(args.packed)
    res = analyze(packed, args.source, tuple(args.splits))
    _print(res)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
