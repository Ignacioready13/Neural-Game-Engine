"""P1 click counterfactual at the H-step horizon (the redesigned probe).

Tests whether a future-H model turns the view toward a *held* door click. For val
contexts with reachable doors on both sides of screen centre, we predict the
``t+H`` frame under three held actions and measure the horizontal image shift
between the last context frame and the prediction (phase correlation; + = content
right = view turned left):

  left  — click the left door, then hold its screenX
  right — click the right door, then hold its screenX
  hold  — no new click (hold the true last-context action)

Three variants isolate *what* the model keys on:

  ``last``       — the click lands on the **final context action** a_{K-1}, and the
                   future is held (deployment-realistic: only the past + the click
                   we inject are known at play time).
  ``future``     — true context, click lands on the first future action a_K.
  ``consistent`` — the same X is held across context AND future (absolute held
                   position, no change event).

Run: .venv/bin/python probe_click.py --packed data/pilot_v1_64x48 \
       --checkpoint runs/<name>/best.pt
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from analysis import load_analysis
from dataset import PackedDataset
from metrics import estimate_shift_x
from model import load_checkpoint

_DOOR_NEAR = 2.0       # world units: both doors must be this close to be "visible"
_NEAR_SPLIT = 0.5      # world units: near/far breakdown threshold


def _select_contexts(packed, sidecar, K, H, max_per_ep=2):
    ep, start = packed.enumerate_windows("val", K, H, require_safe=True)
    seen = {}
    out = []
    for e, s in zip(ep.tolist(), start.tolist()):
        L = packed.episodes[int(e)]["length"]
        if s < 0.1 * L or s > 0.8 * L:          # mid-episode only
            continue
        g = s + K - 1
        cands = [c for c in sidecar.candidates_at(e, g) if c["reachable"]]
        left = [c for c in cands if c["screenX"] < -0.05]
        right = [c for c in cands if c["screenX"] > 0.05]
        if not left or not right:
            continue
        l = min(left, key=lambda c: c["screenDistance"])
        r = min(right, key=lambda c: c["screenDistance"])
        if l["screenDistance"] >= _DOOR_NEAR or r["screenDistance"] >= _DOOR_NEAR:
            continue
        if seen.get(int(e), 0) >= max_per_ep:
            continue
        seen[int(e)] = seen.get(int(e), 0) + 1
        out.append((int(e), int(s), float(l["screenX"]), float(r["screenX"]),
                    float(l["screenDistance"]), float(r["screenDistance"])))
    return out


def _run(packed, model, sidecar, K, H, contexts):
    device = next(model.parameters()).device
    variants = ("last", "future", "consistent")
    res = {v: {"left": [], "right": [], "hold": []} for v in variants}
    meta = []
    model.eval()
    with torch.no_grad():
        for e, s, xl, xr, dl, dr in contexts:
            off = packed.episodes[e]["frame_offset"]
            fr = packed.images[off + s: off + s + K]
            last_u8 = fr[K - 1]
            ctx = torch.from_numpy(np.ascontiguousarray(fr.transpose(0, 3, 1, 2)[None])) \
                .float().to(device) / 127.5 - 1.0
            lab = packed.labels[off + s: off + s + K]
            true_ctx = np.nan_to_num(packed.action_vectors(lab), nan=0.0).astype(np.float32)
            held = true_ctx[K - 1].copy()          # [2] the unchanged held action

            for v in variants:
                for cond, x, clicked in (("left", xl, 1.0), ("right", xr, 1.0),
                                         ("hold", float(held[0]), float(held[1]))):
                    ctx_act = true_ctx.copy()
                    if v in ("last", "consistent"):
                        ctx_act[K - 1] = [x, clicked]
                    if v == "consistent":
                        ctx_act[:] = [x, clicked]
                    fut = np.full((H, 2), [x, clicked], np.float32)
                    full = np.concatenate([ctx_act[None], fut[None]], axis=1)
                    act = torch.from_numpy(full).float().to(device)
                    pred = model.predict(ctx, act)[0]
                    pu = pred.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5
                    pu = (pu * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                    res[v][cond].append(estimate_shift_x(last_u8, pu))

            gt_shift = estimate_shift_x(last_u8, packed.images[off + s + K + H - 1])
            meta.append((e, s, min(dl, dr), gt_shift))
    return res, meta


def _report(res, meta, tag):
    print(f"\n=== P1 click counterfactual — {tag} (n={len(meta)}) ===")
    print(f"distinct episodes: {len({m[0] for m in meta})}")
    labels = {"last": "click@final-context-step", "future": "click@first-future-step",
              "consistent": "held across context+future"}
    for v, lab in labels.items():
        L = np.array(res[v]["left"])
        R = np.array(res[v]["right"])
        Hd = np.array(res[v]["hold"])
        print(f"\n  variant {v} ({lab}):")
        print(f"    mean shift  left {L.mean():+.3f}  right {R.mean():+.3f}  hold {Hd.mean():+.3f}")
        print(f"    left-correct (L>hold) {np.mean(L > Hd):.3f}   "
              f"right-correct (R<hold) {np.mean(R < Hd):.3f}   "
              f"both (L>hold>R) {np.mean((L > Hd) & (Hd > R)):.3f}")
        near = np.array([m[2] < _NEAR_SPLIT for m in meta])
        for label, m in (("near", near), ("far", ~near)):
            if m.sum() == 0:
                continue
            print(f"    [{label} n={m.sum()}] left {np.mean((L > Hd)[m]):.3f}  "
                  f"right {np.mean((R < Hd)[m]):.3f}  both {np.mean(((L > Hd) & (Hd > R))[m]):.3f}")
    gt = np.array([m[3] for m in meta])
    print(f"\n  GT accumulated H-step shift: mean {gt.mean():+.3f}, mean|gt| {np.abs(gt).mean():.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", default="data/pilot_v1_64x48")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-contexts", type=int, default=200)
    ap.add_argument("--max-per-ep", type=int, default=2)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    packed = PackedDataset(args.packed, in_memory=True)
    sidecar = load_analysis(packed)
    model, payload = load_checkpoint(args.checkpoint, device)
    K = payload["model_config"]["K"]
    H = (payload["model_config"].get("n_action_frames") or (K + 10)) - K

    contexts = _select_contexts(packed, sidecar, K, H, max_per_ep=args.max_per_ep)
    print(f"qualifying contexts: {len(contexts)} across "
          f"{len({c[0] for c in contexts})} episodes")
    if len(contexts) > args.max_contexts:
        contexts = contexts[: args.max_contexts]
    res, meta = _run(packed, model, sidecar, K, H, contexts)
    _report(res, meta, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
