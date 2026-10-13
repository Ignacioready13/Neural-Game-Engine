"""Stage 5b diagnosis (Section 5.1 of the overnight brief).

Four questions, answered with numbers, before any iteration is chosen:

  1. FORWARD MOTION — how much of the real frame-to-frame change is radial
     expansion about the vanishing point (walking forward: walls and doorways grow
     outward) versus a pure horizontal shift (turning)?  A translation-only warp can
     never reproduce expansion, so the generator has to invent new content every
     frame; if expansion is a large share of the change, that is where the structure
     is being lost.
  2. REVEALS — how often does the incumbent generator produce a door-like opening,
     compared with real, using the frozen detector?
  3. UNDER-TRAINING / CAPACITY — validation loss history, the denoiser's receptive
     field versus the frame size, and how much of the frame it can actually see.
  4. HARDWARE USE — samples/s, batch size, data-wait share.

Outputs ``eval_out/diagnosis.json`` and prints a short table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from dataset import PackedDataset
from gates import load_model
from judges import OpeningDetector, to_float01, luminance
from metrics import estimate_shift_x

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval_out", "diagnosis.json")


# --------------------------------------------------------------------------- #
# 1. zoom versus shift
# --------------------------------------------------------------------------- #

def _zoom_shift_fit(a_u8, b_u8, margin: int = 12, max_log_zoom: float = 0.06):
    """Fit b ~= zoom(a) shifted horizontally, by brute-force search.

    Returns (log_zoom, shift_px, residual_rel) for the best (zoom, shift), plus the
    residual of the best translation-only warp for comparison.  The residual is
    normalised by the frame's own contrast so it is comparable across clips.
    """
    a = luminance(to_float01(a_u8[None]))[0]
    b = luminance(to_float01(b_u8[None]))[0]
    h, w = a.shape
    a_t = torch.from_numpy(a)[None, None]
    b_t = torch.from_numpy(b)[None, None]

    # central crop of b is held fixed; a is zoomed and shifted to match it
    y0, y1 = margin, h - margin
    x0, x1 = margin, w - margin
    target = b_t[:, :, y0:y1, x0:x1]

    def warp(log_z, shift):
        s = float(np.exp(log_z))
        # sample a at (zoom about the image centre) then translate
        theta = torch.tensor([[[s, 0.0, -shift / (w / 2.0)],
                               [0.0, s, 0.0]]], dtype=torch.float32)
        grid = F.affine_grid(theta, (1, 1, h, w), align_corners=False)
        z = F.grid_sample(a_t, grid, align_corners=False, padding_mode="border")
        return z[:, :, y0:y1, x0:x1]

    best = (0.0, 0.0, 1e9)
    for log_z in np.linspace(0, max_log_zoom, 13):
        for shift in np.arange(-14, 14.5, 1.0):
            d = warp(log_z, shift) - target
            r = float(torch.mean(d ** 2))
            if r < best[2]:
                best = (float(log_z), float(shift), r)
    # refine
    lz, sh, _ = best
    for log_z in np.linspace(max(0.0, lz - 0.01), lz + 0.01, 7):
        for shift in np.arange(sh - 1.5, sh + 1.6, 0.5):
            d = warp(log_z, shift) - target
            r = float(torch.mean(d ** 2))
            if r < best[2]:
                best = (float(log_z), float(shift), r)
    lz, sh, res = best
    contrast = float(torch.mean(target ** 2)) + 1e-8
    return lz, sh, res / contrast


def diagnosis_forward_motion(packed, sets, n_clips=24, device="cpu"):
    """Fit zoom+shift to consecutive real frame pairs; compare with shift only."""
    clips = sets["main_clips"][:n_clips]
    rows = []
    for c in clips:
        e, s = c["episode"], c["start"]
        off = packed.episodes[e]["frame_offset"]
        seq = np.asarray(packed.images[off + s: off + s + 61])
        for t in range(len(seq) - 1):
            lz, sh, res_joint = _zoom_shift_fit(seq[t], seq[t + 1])
            _, _, res_shift = _zoom_shift_fit(seq[t], seq[t + 1], max_log_zoom=0.0005)
            rows.append({"clip": e, "t": t, "log_zoom": lz, "shift_px": sh,
                         "res_joint": res_joint, "res_shift_only": res_shift})
    lz = np.array([r["log_zoom"] for r in rows])
    sh = np.array([r["shift_px"] for r in rows])
    rj = np.array([r["res_joint"] for r in rows])
    rs = np.array([r["res_shift_only"] for r in rows])
    out = {
        "n_pairs": len(rows),
        "mean_log_zoom_per_step": float(lz.mean()),
        "median_log_zoom_per_step": float(np.median(lz)),
        "share_pairs_with_expansion": float((lz > 0.002).mean()),
        "mean_abs_shift_px": float(np.abs(sh).mean()),
        "shift_only_residual": float(rs.mean()),
        "zoom_shift_residual": float(rj.mean()),
        "residual_reduction_from_zoom": float(1.0 - rj.mean() / max(1e-9, rs.mean())),
        "expanded_forward_speed_px_per_step": float(np.exp(lz.mean()) * 0 + 0.0),
    }
    # equivalent forward motion: a log-zoom of L per step is the pinhole signature of
    # moving by roughly (1 - 1/exp(L)) of the distance to the surface in view
    out["forward_motion_fraction_per_step"] = float(1.0 - np.exp(-lz.mean()))
    return out


# --------------------------------------------------------------------------- #
# 3. capacity / receptive field
# --------------------------------------------------------------------------- #

def _receptive_field(model):
    """Receptive field of the denoiser (in input pixels) and of the core."""
    from flow import VelocityUNet

    def rf_of(net):
        rf, stride = 1, 1
        for m in net.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
                s = m.stride[0] if isinstance(m.stride, tuple) else m.stride
                rf = rf + (k - 1) * stride
                stride *= s
        return rf, stride

    info = {}
    for name, mod in (("denoiser", getattr(model, "denoiser", None)),
                      ("core", getattr(model, "core", None))):
        if mod is None:
            continue
        rf, st = rf_of(mod)
        info[name] = {"receptive_field_px": int(rf), "final_stride": int(st),
                      "params": int(sum(p.numel() for p in mod.parameters()))}
    return info


def diagnosis_capacity(ckpt_paths, frame_shape=(72, 96)):
    out = {}
    h, w = frame_shape
    for name, path in ckpt_paths.items():
        if not os.path.exists(path):
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload.get("model_config", {})
        n_params = int(sum(v.numel() for v in payload["state_dict"].values()
                           if hasattr(v, "numel")))
        row = {"n_params_state_dict": n_params,
               "train_steps": cfg.get("max_steps") or payload.get("steps"),
               "history": payload.get("history") or payload.get("val_history")}
        # validation loss trajectory from the training log, if present
        logp = os.path.join(os.path.dirname(path), "train.log")
        if os.path.exists(logp):
            vals = []
            with open(logp) as fh:
                for line in fh:
                    if "val" in line.lower() and "loss" in line.lower():
                        vals.append(line.strip()[:160])
            row["val_log_tail"] = vals[-6:]
        out[name] = row
    return out


# --------------------------------------------------------------------------- #
# 4. hardware use
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 4. hardware use
# --------------------------------------------------------------------------- #

def diagnosis_hardware(model, packed, sets, device="cuda", batches=(64, 128, 256), reps=15):
    """Time a full training step (forward + backward) at several batch sizes.

    Also reports the receptive field, since a failure to invent structure can be a
    receptive-field problem rather than a capacity problem: a denoiser that cannot
    see the vanishing point cannot make a wall expand correctly.
    """
    from flow import flow_training_loss

    K = sets["K"]
    ctx = torch.randn(1, K, 3, 72, 96, device=device)
    feats = torch.randn(1, K, max(4, getattr(model, "feat_dim", 4) or 4), device=device)
    out = {"receptive_field": _receptive_field(model)}
    for bs in batches:
        try:
            c = ctx.repeat(bs, 1, 1, 1, 1)
            f = feats.repeat(bs, 1, 1)
            if c.shape[2] != K:
                c = c.reshape(bs, K, 3, 72, 96)
            f = f.reshape(bs, K, -1)
            target = torch.randn(bs, 3, 72, 96, device=device)
            for _ in range(3):     # warm-up
                loss = flow_training_loss(model, c, f, target)
                loss.backward()
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(reps):
                loss = flow_training_loss(model, c, f, target)
                loss.backward()
                model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            ms = (time.time() - t0) / reps * 1000.0
            out[f"batch_{bs}"] = {"ms_per_step": round(ms, 2),
                                  "samples_per_s": round(bs / (ms / 1000.0), 1),
                                  "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
        except Exception as exc:
            out[f"batch_{bs}"] = {"error": repr(exc)[:200]}
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", default="data/bulk_v1_96x72")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--which", default="all",
                    choices=["all", "motion", "reveals", "capacity", "hardware"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    import scorecard as SC

    sets = SC.load_sets()
    packed = PackedDataset(args.packed)
    rec = {}
    if args.which in ("all", "motion"):
        rec["forward_motion"] = diagnosis_forward_motion(packed, sets)
        print(json.dumps(rec["forward_motion"], indent=1))
    if args.which in ("all", "capacity"):
        rec["capacity"] = diagnosis_capacity({
            "core": "runs/steer96/best.pt", "generator": args.checkpoint})
        print(json.dumps(rec["capacity"], indent=1)[:1500])
    if args.which in ("all", "reveals"):
        det = OpeningDetector.load(os.path.join(HERE, "eval_out", "opening_detector.pt"), "cuda")
        model, _ = load_model(args.checkpoint, "cuda")
        col = SC.collect(packed, model, sets, sets["main_clips"][:12], [0], sets["scenarios"],
                         detector=det, device="cuda")
        ref = SC.real_reference(packed, sets, long_only=True)
        rec["reveals"] = SC.structure(col["bank"], ref["bank"], det, col, device="cuda")
        print(json.dumps(rec["reveals"], indent=1))
    if args.which in ("all", "hardware"):
        model, _ = load_model(args.checkpoint, "cuda")
        rec["capacity_rf"] = _receptive_field(model)
        rec["hardware"] = diagnosis_hardware(model, packed, sets)
        print(json.dumps(rec["capacity_rf"], indent=1))
        print(json.dumps(rec["hardware"], indent=1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        old = json.load(open(args.out))
        old.update(rec)
        rec = old
    with open(args.out, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
