"""Generate the three Stage 4 review images for the user's visual judgment.

  1. predictions.png   — ~8 val windows: last 3 context frames, the ground-truth
                         next frame, the model prediction, and the persistence
                         prediction, with the action values printed.
  2. rollouts.png      — ~3 val clips: ground-truth frames above the model's
                         autoregressive frames (every 4th step for ~40 steps).
  3. counterfactual.png— ~3 contexts with doors on both sides, rolled forward
                         under "click left / right / no click", a few time steps.

Drawn through the loader + the model (never by reading PNGs directly). Uses
matplotlib (Agg); nearest-neighbour so pixels stay crisp when enlarged.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis import load_analysis
from dataset import PackedDataset
from eval_harness import _torch_to_u8, rollout
from model import load_checkpoint

_HEADING_EPS = np.deg2rad(0.5)


def _action_str(a):
    ax = a[0]
    return "null" if np.isnan(ax) else f"{ax:+.2f}"


def _select_turn_windows(packed, sidecar, K, n_turn, n_steady):
    """Return (turning windows, steady windows) as (episode, start) lists."""
    ep, start = packed.enumerate_windows("val", K, 1, require_safe=True)
    turns, steady = [], []
    for e, s in zip(ep.tolist(), start.tolist()):
        d = sidecar.heading_change(e, s + K - 1)
        if np.isnan(d):
            continue
        if abs(d) >= _HEADING_EPS:
            turns.append((abs(d), (e, s)))
        else:
            steady.append((abs(d), (e, s)))
    turns.sort(reverse=True)
    steady.sort()
    # de-duplicate: keep one window per episode so the image shows variety
    pick_turn, pick_steady = [], []
    for _, (e, s) in turns:
        if e not in {p[0] for p in pick_turn}:
            pick_turn.append((e, s))
        if len(pick_turn) >= n_turn:
            break
    for _, (e, s) in steady:
        if e not in {p[0] for p in pick_steady}:
            pick_steady.append((e, s))
        if len(pick_steady) >= n_steady:
            break
    return pick_turn, pick_steady


def make_predictions_image(packed, model, K, out_path, sidecar, n_turn=6, n_steady=2):
    turns, steady = _select_turn_windows(packed, sidecar, K, n_turn, n_steady)
    windows = [(e, s, "turn") for e, s in turns] + [(e, s, "steady") for e, s in steady]
    if not windows:
        raise RuntimeError("no val windows to draw")

    ncols = 3 + 3  # 3 context + GT + model + persistence
    fig, axs = plt.subplots(len(windows), ncols,
                            figsize=(ncols * 1.5, len(windows) * 1.7))
    dev = next(model.parameters()).device
    for r, (e, s, kind) in enumerate(windows):
        w = packed.window(e, s, K, 1)
        ctx = np.ascontiguousarray(w.context)
        ctx_t = torch.from_numpy(ctx.transpose(0, 3, 1, 2)[None]).float().to(dev) / 127.5 - 1.0
        act = torch.from_numpy(np.ascontiguousarray(w.context_action)[None]).float().to(dev)
        with torch.no_grad():
            pred = _torch_to_u8(model.predict(ctx_t, act)[0])
        pers = w.frames[K - 1]
        gt = w.frames[K]

        panels = [w.frames[K - 3], w.frames[K - 2], w.frames[K - 1], gt, pred, pers]
        labels = [f"ctx-2\naX={_action_str(w.actions[K-3])}",
                  f"ctx-1\naX={_action_str(w.actions[K-2])}",
                  f"ctx\n(last)\naX={_action_str(w.actions[K-1])}",
                  "GT\nnext", "MODEL\npred", "PERSIST\n=last ctx"]
        for c, (img, lab) in enumerate(zip(panels, labels)):
            axs[r, c].imshow(img, interpolation="nearest")
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
            axs[r, c].set_title(lab, fontsize=8)
            if c >= 3:
                for sp in axs[r, c].spines.values():
                    sp.set_color("black" if c == 3 else "blue")
                    sp.set_linewidth(2)
        axs[r, 0].set_ylabel(f"{kind}\nep{e} s{s}", fontsize=8)
    fig.suptitle("predictions.png — context -> next frame (model vs persistence)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def make_rollouts_image(packed, model, K, out_path, steps=40, stride=4, n_clips=3):
    # pick 3 val clips with room for the rollout
    ep, start = packed.enumerate_windows("val", K, 1, require_safe=True)
    clips = []
    for e, s in zip(ep.tolist(), start.tolist()):
        if s + K + steps <= packed.episodes[int(e)]["length"]:
            clips.append((e, s))
        if len(clips) >= n_clips:
            break
    cols = list(range(0, steps, stride))
    fig, axs = plt.subplots(len(clips) * 2, len(cols),
                            figsize=(len(cols) * 1.4, len(clips) * 2 * 1.7))
    for ci, (e, s) in enumerate(clips):
        preds = rollout(packed, model, e, s, K, steps, device="cuda")
        preds_u8 = _torch_to_u8(preds)  # [steps,H,W,3]
        off = packed.episodes[e]["frame_offset"]
        gt = packed.images[off + s + K: off + s + K + steps]
        for cj, t in enumerate(cols):
            axs[2 * ci, cj].imshow(gt[t], interpolation="nearest")
            axs[2 * ci + 1, cj].imshow(preds_u8[t], interpolation="nearest")
            axs[2 * ci, cj].set_xticks([]); axs[2 * ci, cj].set_yticks([])
            axs[2 * ci + 1, cj].set_xticks([]); axs[2 * ci + 1, cj].set_yticks([])
            axs[2 * ci, cj].set_title(f"t={t}", fontsize=8)
        axs[2 * ci, 0].set_ylabel(f"GT ep{e} s{s}", fontsize=8)
        axs[2 * ci + 1, 0].set_ylabel("MODEL", fontsize=8)
    fig.suptitle("rollouts.png — top: ground truth, bottom: model autoregression "
                 "(every 4th step, true actions)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def make_counterfactual_image(packed, model, K, out_path, sidecar, steps=15,
                              n_contexts=3, eps=0.05, show_steps=(1, 4, 8, 12, 15)):
    ep, start = packed.enumerate_windows("val", K, 1, require_safe=True)
    contexts = []
    for e, s in zip(ep.tolist(), start.tolist()):
        g = s + K - 1
        if g + 1 + steps > packed.episodes[int(e)]["length"]:
            continue
        cands = [c for c in sidecar.candidates_at(e, g) if c["reachable"]]
        xs = [c["screenX"] for c in cands]
        left = [x for x in xs if x < -eps]
        right = [x for x in xs if x > eps]
        if left and right:
            contexts.append((e, s, min(left), max(right)))
        if len(contexts) >= n_contexts:
            break

    rows = [("LEFT (click)", True), ("RIGHT (click)", True), ("NONE", False)]
    nrows = len(contexts) * len(rows)
    fig, axs = plt.subplots(nrows, len(show_steps),
                            figsize=(len(show_steps) * 1.5, nrows * 1.7))
    if nrows == 1:
        axs = axs[None, :]
    dev = next(model.parameters()).device
    for ci, (e, s, xl, xr) in enumerate(contexts):
        for ri, (lab, clicked) in enumerate(rows):
            x = (xl if ri == 0 else xr) if ri < 2 else 0.0
            a = torch.from_numpy(np.full((1, K, 2), [x, 1.0 if clicked else 0.0],
                                         dtype=np.float32)).to(dev)
            with torch.no_grad():
                preds = _torch_to_u8(rollout(packed, model, e, s, K, steps, a, dev))
            for cj, t in enumerate(show_steps):
                r = ci * len(rows) + ri
                axs[r, cj].imshow(preds[t - 1], interpolation="nearest")
                axs[r, cj].set_xticks([]); axs[r, cj].set_yticks([])
                axs[r, cj].set_title(f"t={t}", fontsize=8)
            axs[ci * len(rows) + ri, 0].set_ylabel(
                f"ep{e} {lab} aX={x:+.2f}", fontsize=8)
    fig.suptitle("counterfactual.png — same context rolled under left/right/no click",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    packed = PackedDataset(args.packed, in_memory=True)
    model, payload = load_checkpoint(args.checkpoint, device)
    K = payload["model_config"]["K"]
    sidecar = load_analysis(packed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("predictions:", make_predictions_image(
        packed, model, K, os.path.join(args.out_dir, "predictions.png"), sidecar))
    print("rollouts:", make_rollouts_image(
        packed, model, K, os.path.join(args.out_dir, "rollouts.png")))
    print("counterfactual:", make_counterfactual_image(
        packed, model, K, os.path.join(args.out_dir, "counterfactual.png"), sidecar))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
