"""Stage 5b review images (brief §10).

The Stage 5 review figures were small with large empty areas, so this module draws
tight grids instead: every frame is rendered at ``--scale`` times its native size,
axes are switched off, row labels sit in a narrow left gutter, and the figure is
saved with ``bbox_inches="tight"`` and almost no padding.

Images produced (each as its own PNG so nothing is shrunk to fit):
  rollouts_v0.png / rollouts_<final>.png   real above generated, 4 clips, every
                                           4th step to 60
  counterfactual_<name>.png                left / right / no-click, 4 episodes
  reveals_<name>.png                       real, Stage 4 baseline, 4 seeds
  longrun_<name>.png                       a frame every 20 steps over 300 steps,
                                           no-click and scripted, v0 above the model
  candidate_<name>_strip.png               one long-run strip per other candidate
  play_<name>.gif                          a short GIF (via play.py)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

from dataset import PackedDataset, frames_to_tensor   # noqa: E402
from eval_harness import features_from_history, model_predict  # noqa: E402
from gates import load_model                          # noqa: E402
from scorecard import _u8, _script_for, load_sets      # noqa: E402
from play import write_gif                            # noqa: E402


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #

def _fig_size(n_cols, n_rows, scale):
    """Figure size so each frame really is ``scale`` x its native pixel size."""
    w, h = 96, 72
    base_dpi = 100.0
    fw = n_cols * (w * scale) / base_dpi
    fh = n_rows * (h * scale) / base_dpi
    return fw, fh


def grid(frames_rows, path, scale=3, row_labels=None, col_labels=None, title=None,
         dpi=100, label_width_frac=0.055):
    """Draw a tight grid of uint8 frames.

    ``frames_rows`` is a list of rows; each row is a list of (H,W,3) uint8 frames.
    """
    n_rows = len(frames_rows)
    n_cols = max(len(r) for r in frames_rows)
    fw, fh = _fig_size(n_cols, n_rows, scale)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fw, fh), dpi=dpi,
                             gridspec_kw={"wspace": 0.01, "hspace": 0.02,
                                          "left": label_width_frac, "right": 1.0,
                                          "top": 1.0 if not title else 0.94,
                                          "bottom": 0.0})
    axes = np.atleast_2d(axes)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i][j]
            ax.set_axis_off()
            if j < len(frames_rows[i]):
                ax.imshow(frames_rows[i][j], interpolation="nearest")
    if row_labels:
        for i, lab in enumerate(row_labels):
            axes[i][0].text(-0.03, 0.5, lab, transform=axes[i][0].transAxes,
                            ha="right", va="center", fontsize=9, rotation=0)
    if col_labels:
        for j, lab in enumerate(col_labels):
            axes[0][j].set_title(lab, fontsize=8, pad=2)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# rollouts
# --------------------------------------------------------------------------- #

def _rollout_sequence(packed, model, e, s, K, steps, scenario, seed, device):
    rec = packed.episodes[e]
    off = rec["frame_offset"]
    hist = [r for r in np.nan_to_num(packed.action_vectors(
        packed.labels[off + max(0, s - 8): off + s + K]), nan=0.0).astype(np.float32)]
    frames = frames_to_tensor(np.asarray(packed.images[off + s: off + s + K])).to(device)
    script = dict(_script_for(steps)) if scenario == "scripted_alt" else {}
    held = (0.0, 0.0)
    release = -1
    out = [np.asarray(packed.images[off + s + k]) for k in range(K)]
    with torch.no_grad():
        for t in range(steps):
            if scenario == "real_click":
                a = np.nan_to_num(packed.action_vectors(
                    packed.labels[off + s + t: off + s + t + K]), nan=0.0).astype(np.float32)
                hist.extend([a[-1]])
            else:
                if scenario == "scripted_alt":
                    if t in script:
                        held = (float(script[t]), 1.0)
                        release = t + 3
                    elif t >= release:
                        held = (held[0], 0.0)
                a = np.tile(np.array(held, np.float32), (K, 1))
                hist.append(np.array(held, np.float32))
            feats = features_from_history(np.stack(hist), K, device)
            a_t = torch.from_numpy(np.ascontiguousarray(a))[None].to(device)
            nxt = model_predict(model, frames[-K:].unsqueeze(0), a_t, feats, seed)
            nxt = nxt[0] if isinstance(nxt, tuple) else nxt
            frames = torch.cat([frames, nxt], dim=0)
            out.append(_u8(nxt[0]))
    return out[K - 1:], out[:K]


def make_rollouts(packed, models, sets, out_dir, every=4, upto=60, clips=None,
                  device="cuda", scale=3):
    K = sets["K"]
    clips = clips or sets["main_clips"][:4]
    paths = []
    for ci, clip in enumerate(clips):
        e, s = clip["episode"], clip["start"]
        rows, labels = [], []
        rec = packed.episodes[e]
        off = rec["frame_offset"]
        # the real row: the clip's own frames at the same steps
        real_row = [np.asarray(packed.images[off + s + K - 1 + i])
                    for i in range(0, upto + 1, every)]
        idx = list(range(0, upto + 1, every))
        rows.append(real_row)
        labels.append("real")
        for name, m in models:
            seq, _ = _rollout_sequence(packed, m, e, s, K, upto, "real_click", 0, device)
            rows.append([seq[i] if i < len(seq) else seq[-1] for i in idx])
            labels.append(name)
        p = os.path.join(out_dir, f"rollouts_clip{ci}_{e}.png")
        grid(rows, p, scale=scale, row_labels=labels,
             col_labels=[str(i) for i in idx],
             title=f"clip {ci}: episode {e} start {s} — real clicks, "
                   f"every {every} steps to {upto} (column = step)")
        paths.append(p)
    return paths


def make_counterfactual(packed, models, sets, out_dir, device="cuda", scale=3):
    K = sets["K"]
    paths = []
    for ci, clip in enumerate(sets["main_clips"][4:8]):
        e, s = clip["episode"], clip["start"]
        rows, labels = [], []
        for name, m in models:
            r = []
            for x in (-0.8, 0.8, None):
                rec = packed.episodes[e]
                off = rec["frame_offset"]
                hist = [r2 for r2 in np.nan_to_num(packed.action_vectors(
                    packed.labels[off + max(0, s - 8): off + s + K]), nan=0.0).astype(np.float32)]
                frames = frames_to_tensor(
                    np.asarray(packed.images[off + s: off + s + K])).to(device)
                held = (0.0, 0.0) if x is None else (float(x), 1.0)
                with torch.no_grad():
                    for t in range(24):
                        hist.append(np.array(held, np.float32))
                        feats = features_from_history(np.stack(hist), K, device)
                        a_t = torch.from_numpy(np.tile(np.array(held, np.float32), (K, 1)))[None].to(device)
                        nxt = model_predict(m, frames[-K:].unsqueeze(0), a_t, feats, 0)
                        nxt = nxt[0] if isinstance(nxt, tuple) else nxt
                        frames = torch.cat([frames, nxt], dim=0)
                r.append(_u8(nxt[0]))
            rows.append(r)
            labels.append(name)
        p = os.path.join(out_dir, f"counterfactual_{ci}_{e}.png")
        grid(rows, p, scale=scale, row_labels=labels,
             col_labels=["click -0.8", "click +0.8", "no click"],
             title=f"counterfactual, clip {ci} (episode {e}), 24 generated steps")
        paths.append(p)
    return paths


def make_reveals(packed, final_model, sets, out_dir, device="cuda", scale=3,
                 baseline_ckpt=None, seeds=(0, 1, 2, 3), steps=24):
    """At real doorway-reveal windows: real, Stage 4 baseline, and 4 seeds.

    One figure per reveal window: stacking all of them into a single image made a
    872x5084 strip that is unreadable in any normal viewer.
    """
    from gates import reveal_and_steady_windows

    K = sets["K"]
    hi, _ = reveal_and_steady_windows(packed, "val", K, n=6)
    hi = [(e, st) for _, e, st in hi]      # [(frame_change, episode, start)]
    base = load_model(baseline_ckpt, device)[0] if baseline_ckpt else None
    paths = []
    for wi, (e, s) in enumerate(hi[:4]):
        rec = packed.episodes[e]
        off = rec["frame_offset"]
        rows, labels = [], []
        rows.append([np.asarray(packed.images[off + s + K - 1 + j]) for j in (0, 8, 16)])
        labels.append("real")
        if base is not None:
            seq, _ = _rollout_sequence(packed, base, e, s, K, steps, "real_click", 0, device)
            rows.append([seq[7], seq[15], seq[steps - 1]])
            labels.append("Stage 4")
        for sd in seeds:
            seq, _ = _rollout_sequence(packed, final_model, e, s, K, steps, "real_click",
                                       sd, device)
            rows.append([seq[7], seq[15], seq[steps - 1]])
            labels.append(f"seed {sd}")
        p = os.path.join(out_dir, f"reveals_{wi}_ep{e}.png")
        grid(rows, p, scale=scale, row_labels=labels, col_labels=["+8", "+16", f"+{steps}"],
             title=f"doorway-reveal window {wi} (episode {e}): real, Stage 4 baseline, "
                   f"then four sampling seeds of the final model")
        paths.append(p)
    return paths


def make_longrun(packed, models, sets, out_dir, device="cuda", scale=2, steps=300,
                 every=20):
    """A frame every 20 steps over 300 steps, no-click and scripted; v0 above final."""
    K = sets["K"]
    paths = []
    rows, labels = [], []
    for sc in ("no_click", "scripted_alt"):
        clip = sets["long_clips"][0]
        for name, m in models:
            seq, _ = _rollout_sequence(packed, m, clip["episode"], clip["start"], K,
                                       steps, sc, 0, device)
            rows.append([seq[i - 1] if i > 0 else seq[0]
                         for i in range(0, steps + 1, every)])
            labels.append(f"{name} / {sc}")
        p = os.path.join(out_dir, f"longrun_{sc}.png")
        grid(rows, p, scale=scale, row_labels=labels,
             col_labels=[str(i) for i in range(0, steps + 1, every)],
             title=f"long run ({sc}), frame every {every} steps to {steps} — "
                   f"rows: incumbent v0 then the candidate")
        paths.append(p)
        rows, labels = [], []
    return paths


def make_candidate_strip(packed, model, sets, out_dir, name, device="cuda", scale=2,
                         steps=300, every=20):
    K = sets["K"]
    rows, labels = [], []
    for sc in ("no_click", "scripted_alt"):
        clip = sets["long_clips"][1]
        seq, _ = _rollout_sequence(packed, model, clip["episode"], clip["start"], K,
                                   steps, sc, 0, device)
        rows.append([seq[i - 1] if i > 0 else seq[0]
                     for i in range(0, steps + 1, every)])
        labels.append(sc)
    p = os.path.join(out_dir, f"candidate_{name}_strip.png")
    grid(rows, p, scale=scale, row_labels=labels,
         col_labels=[str(i) for i in range(0, steps + 1, every)],
         title=f"candidate '{name}': 300-step long run, every {every} steps")
    return p


def make_gif(packed, model, sets, out_dir, name, device="cuda", frames=200):
    K = sets["K"]
    clip = sets["main_clips"][0]
    seq, _ = _rollout_sequence(packed, model, clip["episode"], clip["start"], K,
                               frames, "scripted_alt", 0, device)
    p = os.path.join(out_dir, f"play_{name}.gif")
    write_gif(seq, p, scale=4, fps=20, max_frames=frames)
    return p


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", default="data/bulk_v1_96x72")
    ap.add_argument("--v0", default="runs/flow96/best.pt")
    ap.add_argument("--final", required=True)
    ap.add_argument("--baseline", default="runs/baseline_96/best.pt")
    ap.add_argument("--out", default="review/stage5b")
    ap.add_argument("--which", default="all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scale", type=int, default=3)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    sets = load_sets()
    packed = PackedDataset(args.packed)
    v0 = load_model(args.v0, args.device)[0]
    fin = load_model(args.final, args.device)[0]
    final_name = os.path.basename(os.path.dirname(args.final))
    models = [("v0", v0), (final_name, fin)]
    made = []
    todo = args.which.split(",")
    if "all" in todo or "rollouts" in todo:
        made += make_rollouts(packed, models, sets, args.out, device=args.device,
                              scale=args.scale)
    if "all" in todo or "counterfactual" in todo:
        made += make_counterfactual(packed, models, sets, args.out, device=args.device,
                                    scale=args.scale)
    if "all" in todo or "reveals" in todo:
        made += make_reveals(packed, fin, sets, args.out, device=args.device,
                             scale=args.scale, baseline_ckpt=args.baseline)
    if "all" in todo or "longrun" in todo:
        made += make_longrun(packed, models, sets, args.out, device=args.device,
                             scale=args.scale)
    if "all" in todo or "strip" in todo:
        made.append(make_candidate_strip(packed, fin, sets, args.out, final_name,
                                         device=args.device, scale=args.scale))
    if "all" in todo or "gif" in todo:
        made.append(make_gif(packed, fin, sets, args.out, final_name,
                             device=args.device))
    for p in made:
        print(f"wrote {p}  ({os.path.getsize(p) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
