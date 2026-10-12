"""Stage 5 — review images for the user (brief §6).

Renders the artifacts the brief asks for, all drawn *through the loader and the
model* (never by reading PNGs):

  1. ``rollouts.png``      4 val clips, real row above the generated row, every
                           4th step for 60 steps, true clicks supplied.
  2. ``counterfactual.png`` 4 contexts with a close door on each side, rolled under
                           "click the left door" / "right" / "no click".
  3. ``reveals.png``      4 doorway-reveal moments: the real next frames, the Stage 4
                           deterministic baseline, and 4 generator seeds.
  4. ``longrun.png``      (Milestone 3) 2 closed-loop 300-frame rollouts.
  5. GIFs of closed-loop rollouts.

Visual judgement is the user's; this script only prepares the images.

Run::

    ./memcap.sh --mem 12G -- .venv/bin/python review_stage5.py \
        --packed data/bulk_v1_96x72 --checkpoint runs/flow96/best.pt \
        --out-dir review/stage5 --which all
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

from analysis import load_analysis
from dataset import PackedDataset, frames_to_tensor
from eval_harness import model_predict
from gates import (_rollout_seq, _u8, load_model, near_door_contexts,
                   onset_windows)
from metrics import frame_change
from shift_table import load_target_table


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #

def _cell(frame_u8, scale):
    im = Image.fromarray(frame_u8)
    return im.resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST)


def _label_row(draw, x, y, w, text, height=16, bg=(25, 25, 30), fg=(230, 230, 230)):
    draw.rectangle([x, y, x + w, y + height], fill=bg)
    draw.text((x + 4, y + 2), text, fill=fg)


def _grid(rows, path, scale=3, caption_h=16, sep=6, row_labels=None):
    """``rows`` is a list of lists of ``(frame_u8, label)`` cells; one row per list."""
    cells = [[_cell(f, scale) for f, _ in row] for row in rows]
    cw, ch = cells[0][0].size
    ncol = max(len(r) for r in cells)
    W = ncol * (cw + sep) + sep
    H = sum(len(r) * (ch + caption_h + sep) + sep for r in rows)
    sheet = Image.new("RGB", (W, H), (12, 12, 14))
    draw = ImageDraw.Draw(sheet)
    y = sep
    for ri, row in enumerate(cells):
        if row_labels and row_labels[ri]:
            _label_row(draw, 0, y, W, row_labels[ri], height=caption_h + 4,
                       bg=(35, 35, 45))
            y += caption_h + 4
        x = sep
        labels = [lab for _, lab in rows[ri]]
        for ci, im in enumerate(row):
            sheet.paste(im, (x, y))
            if labels[ci]:
                draw.text((x + 2, y + ch + 1), labels[ci], fill=(200, 220, 200))
            x += cw + sep
        y += ch + caption_h + sep
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sheet.save(path)
    return path, sheet.size


def _act_label(a):
    return f"aX={a[0]:+.2f}{'*' if a[1] > 0.5 else ' '}"


# --------------------------------------------------------------------------- #
# 1. rollouts
# --------------------------------------------------------------------------- #

def render_rollouts(packed, model, out, K=4, steps=60, every=4, n_clips=4, scale=2,
                    device="cuda"):
    ep, start = packed.enumerate_windows("val", K, 1, require_safe=True)
    keep = [(int(e), int(s)) for e, s in zip(ep.tolist(), start.tolist())
            if s + K + steps + 2 < packed.episodes[int(e)]["length"]]
    idx = np.linspace(0, len(keep) - 1, n_clips).astype(int)
    flat, rl = [], []
    for i in idx:
        e, s = keep[i]
        seq, _ = _rollout_seq(packed, model, e, s, K, steps, seed=0, device=device)
        real = _real_seq_padded(packed, e, s, K, steps)
        n = min(len(seq), len(real))
        js = list(range(0, n, every))
        flat.append([(real[j], f"f{j}") for j in js])
        rl.append(f"REAL      ep{e} frames {s + K - 1}..{s + K - 1 + n - 1}  (true clicks)")
        flat.append([(seq[j], f"f{j}") for j in js])
        rl.append(f"GENERATED ep{e}  seed=0   (4-step flow)")
    return _grid(flat, out, scale=scale, row_labels=rl)[0]


def _real_seq_padded(packed, e, s, K, steps):
    rec = packed.episodes[e]
    off, L = rec["frame_offset"], rec["length"]
    lo = s + K - 1
    hi = min(lo + steps + 1, L)
    return np.asarray(packed.images[off + lo: off + hi])


# --------------------------------------------------------------------------- #
# 2. counterfactual
# --------------------------------------------------------------------------- #

def render_counterfactual(packed, sidecar, model, out, K=4, steps=40, n=4, scale=3,
                          device="cuda", every=8):
    ctxs, _ = near_door_contexts(packed, sidecar, K, 10, max_per_ep=1)
    idx = np.linspace(0, len(ctxs) - 1, n).astype(int)
    rows, rl = [], []
    for i in idx:
        e, s, xl, xr, dl, dr = ctxs[i]
        rec = packed.episodes[e]
        off = rec["frame_offset"]
        hist = packed.action_vectors(packed.labels[off + max(0, s - 8): off + s + K])
        held = np.nan_to_num(hist, nan=0.0).astype(np.float32)[-1]
        seed_row = [(np.asarray(packed.images[off + s + K - 1]), "context")]
        for cond, x, clicked in ((f"click L door aX={xl:+.2f}", xl, 1.0),
                                 (f"click R door aX={xr:+.2f}", xr, 1.0),
                                 (f"no click aX={held[0]:+.2f}", float(held[0]),
                                  float(held[1]))):
            ov = torch.tensor([[[x, clicked]] * K], dtype=torch.float32).to(device)
            seq, _ = _rollout_seq(packed, model, e, s, K, steps, override=ov, seed=0,
                                  device=device)
            row = list(seed_row)
            for j in range(every, min(steps, len(seq) - 1) + 1, every):
                row.append((seq[j], _act_label((x, clicked))))
            row.append((seq[-1], "t=40"))
            rows.append(row)
            rl.append(f"ep{e} start s={s}  doorL dist {dl:.2f} doorR dist {dr:.2f}  |  {cond}")
    return _grid(rows, out, scale=scale, row_labels=rl)[0]


# --------------------------------------------------------------------------- #
# 3. reveals
# --------------------------------------------------------------------------- #

@torch.no_grad()
def render_reveals(packed, model, out, baseline_ckpt=None, K=4, n=4, seeds=(0, 1, 2, 3),
                   scale=3, device="cuda"):
    ep, start = packed.enumerate_windows("val", K, 1, require_safe=True)
    changes = []
    rng = np.random.default_rng(0)
    cand = np.arange(len(ep))
    if len(cand) > 4000:
        cand = np.sort(rng.choice(cand, 4000, replace=False))
    for i in cand.tolist():
        e, s = int(ep[i]), int(start[i])
        rec = packed.episodes[e]
        off = rec["frame_offset"]
        prev = np.asarray(packed.images[off + s + K - 1])
        nxt = np.asarray(packed.images[off + s + K])
        changes.append((frame_change(prev, nxt), e, s))   # metrics accept uint8
    changes.sort(reverse=True)
    seen, picked = set(), []
    for c, e, s in changes:
        if e in seen:
            continue
        seen.add(e)
        picked.append((e, s))
        if len(picked) >= n:
            break
    baseline = None
    if baseline_ckpt and os.path.exists(baseline_ckpt):
        baseline, _ = load_model(baseline_ckpt, device)
    rows, rl = [], []
    for e, s in picked:
        rec = packed.episodes[e]
        off = rec["frame_offset"]
        real = [np.asarray(packed.images[off + s + K + j]) for j in range(K)]
        row = [(np.asarray(packed.images[off + s + K - 1]), "context (real)")]
        row += [(f, f"real t+{j + 1}") for j, f in enumerate(real[:2])]
        if baseline is not None:
            try:
                from eval_harness import features_from_history
                lo = max(0, s - 8)
                h = np.nan_to_num(packed.action_vectors(
                    packed.labels[off + lo: off + s + K]), nan=0.0).astype(np.float32)
                ctx = frames_to_tensor(np.asarray(
                    packed.images[off + s: off + s + K]))[None].to(device)
                a = torch.from_numpy(h[-K:])[None].to(device)
                b = model_predict(baseline, ctx, a)[0]
                row.append((_u8(b), "stage4 baseline"))
            except Exception as exc:  # pragma: no cover - diagnostic only
                print("baseline row skipped:", exc)
        ctx = frames_to_tensor(np.asarray(
            packed.images[off + s: off + s + K]))[None].to(device)
        lo = max(0, s - 8)
        h = np.nan_to_num(packed.action_vectors(
            packed.labels[off + lo: off + s + K]), nan=0.0).astype(np.float32)
        feats = click_features_np(h, K, device)
        for sd in seeds:
            p = model_predict(model, ctx, None, feats, sd)
            p = p[0] if isinstance(p, tuple) else p
            row.append((_u8(p[0]), f"gen seed={sd}"))
        rows.append(row)
        rl.append(f"reveal ep{e} s={s}")
    return _grid(rows, out, scale=scale, row_labels=rl)[0]


def click_features_np(hist, K, device):
    from steer import click_features
    return click_features(torch.from_numpy(np.ascontiguousarray(hist))[None]
                          .float().to(device))[:, -K:]


# --------------------------------------------------------------------------- #
# 4. long runs (Milestone 3)
# --------------------------------------------------------------------------- #

def render_longrun(packed, model, out, K=4, steps=300, every=20, seed=0, device="cuda"):
    ep, start = packed.enumerate_windows("val", K, 1, require_safe=True)
    keep = [(int(e), int(s)) for e, s in zip(ep.tolist(), start.tolist())
            if s + K + steps + 2 < packed.episodes[int(e)]["length"]]
    rows, rl = [], []
    # (a) no clicks at all
    e, s = keep[0]
    ov = torch.zeros(1, K, 2, dtype=torch.float32, device=device)
    seq, _ = _rollout_seq(packed, model, e, s, K, steps, override=ov, seed=seed,
                          device=device)
    real = _real_seq_padded(packed, e, s, K, steps)
    n = min(len(seq), len(real))
    rows.append([(real[j], f"real {j}") for j in range(0, n, every)])
    rows.append([(seq[j], f"gen {j}") for j in range(1, n, every)])
    rl.append(f"A) no clicks — ep{e} start {s}")
    rl.append("A) generated, no clicks")
    # (b) scripted alternating clicks
    e2, s2 = keep[len(keep) // 2]
    seq2 = _scripted_rollout(packed, model, e2, s2, K, steps, device, seed)
    rows.append([(seq2[j], f"gen {j}") for j in range(1, n, every)])
    rl.append(f"B) scripted alternating clicks — ep{e2} start {s2}")
    return _grid(rows, out, scale=2, row_labels=rl)[0]


@torch.no_grad()
def _scripted_rollout(packed, model, e, s, K, steps, device, seed):
    from eval_harness import features_from_history
    from steer import click_features
    rec = packed.episodes[e]
    off = rec["frame_offset"]
    lo = max(0, s - 8)
    hist = [r for r in np.nan_to_num(packed.action_vectors(
        packed.labels[off + lo: off + s + K]), nan=0.0).astype(np.float32)]
    frames = frames_to_tensor(np.asarray(packed.images[off + s: off + s + K])).to(device)
    held = (0.0, 0.0)
    out_frames = []
    for t in range(steps):
        if t % 40 == 0:
            held = (0.8 if (t // 40) % 2 == 0 else -0.8, 1.0)
        hist.append(np.array(held, dtype=np.float32))
        feats = click_features(torch.from_numpy(np.stack(hist))[None].float()
                               .to(device))[:, -K:]
        nxt = model_predict(model, frames[-K:].unsqueeze(0), None, feats, seed)
        nxt = nxt[0] if isinstance(nxt, tuple) else nxt
        # keep only a rolling window on the GPU; the full history is not needed
        frames = torch.cat([frames, nxt], dim=0)[-(K + 2):]
        out_frames.append(nxt[0])
    return [_u8(f) for f in out_frames]


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 5 review images")
    ap.add_argument("--packed", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", dest="out_dir", default="review/stage5")
    ap.add_argument("--which", default="all", choices=["all", "1", "2", "3", "4", "5"])
    ap.add_argument("--baseline-64", dest="baseline_64", default="runs/baseline_64/best.pt")
    ap.add_argument("--baseline-96", dest="baseline_96", default="runs/baseline_96/best.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    packed = PackedDataset(args.packed)
    model, payload = load_model(args.checkpoint, device)
    K = int(payload.get("model_config", {}).get("K", 4))
    os.makedirs(args.out_dir, exist_ok=True)
    want = {args.which} if args.which != "all" else {"1", "2", "3", "4", "5"}

    if "1" in want:
        p = render_rollouts(packed, model, os.path.join(args.out_dir, "rollouts.png"),
                            K=K, device=device)
        print("wrote", p, flush=True)
    if "2" in want:
        sidecar = load_analysis(packed)
        p = render_counterfactual(packed, sidecar, model,
                                  os.path.join(args.out_dir, "counterfactual.png"),
                                  K=K, device=device)
        print("wrote", p, flush=True)
    if "3" in want:
        bl = args.baseline_64 if packed.width == 64 else args.baseline_96
        p = render_reveals(packed, model, os.path.join(args.out_dir, "reveals.png"),
                           baseline_ckpt=bl, K=K, device=device)
        print("wrote", p, flush=True)
    if "4" in want:
        p = render_longrun(packed, model, os.path.join(args.out_dir, "longrun.png"),
                           K=K, device=device)
        print("wrote", p, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
