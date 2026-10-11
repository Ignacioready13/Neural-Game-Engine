"""The Stage 4 evaluation harness (the long-lived part).

Evaluates ANY model that has the common interface ``predict(context, action) ->
next_frame`` (context ``[B,K,3,H,W]`` in [-1,1], action ``[B,K,2]``, next
``[B,3,H,W]`` in [-1,1]), plus an autoregressive rollout driver. It is split into
the five parts of brief section 2.4:

  (a) one-step (teacher-forced) metrics, overall and on an "action-relevant"
      subset (windows whose target-step heading change is noticeable);
  (b) action-use ablation WITHOUT retraining: true / zeroed / shuffled / shifted
      actions, overall and action-relevant;
  (c) autoregressive rollouts: error vs horizon for the model and persistence;
  (d) action-following: does the model's inter-frame image shift agree with the
      real frames' shift and with the sidecar heading change?
  (e) counterfactual direction: roll a context under "click left door / right
      door / no click" and test whether the view turns toward the clicked door.

Results are returned as a plain dict and can be written to JSON. The reference
"persistence" predictor (repeat the last context frame) is built in.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from analysis import Analysis, load_analysis
from dataset import PackedDataset, frames_to_tensor
from metrics import estimate_shift_x, l1, mse, psnr, ssim
from model import PersistenceModel, load_checkpoint

# Windows whose target-step heading change magnitude is at least this many degrees
# count as "action-relevant" (a turn is happening). 0.5 deg/frame at 20 Hz = 10 deg/s.
ACTION_RELEVANT_DEG = 0.5
_HEADING_EPS = np.deg2rad(ACTION_RELEVANT_DEG)


# --------------------------------------------------------------------------- #
# Batch helpers
# --------------------------------------------------------------------------- #

def _make_batch(packed, ep, start, K, device):
    """Windows (episode, start) -> (ctx, act, tgt) normalized tensors on `device`."""
    ctx, act, tgt = [], [], []
    for e, s in zip(ep.tolist(), start.tolist()):
        w = packed.window(int(e), int(s), K, 1)
        ctx.append(np.ascontiguousarray(w.context))          # [K, h, w, 3]
        act.append(np.ascontiguousarray(w.context_action))   # [K, 2]
        tgt.append(np.ascontiguousarray(w.target[0]))        # [h, w, 3]
    ctx = np.stack(ctx).transpose(0, 1, 4, 2, 3)             # [B, K, 3, h, w]
    ctx = torch.from_numpy(np.ascontiguousarray(ctx)).float().to(device) / 127.5 - 1.0
    act = torch.from_numpy(np.stack(act)).float().to(device)
    tgt = np.stack(tgt).transpose(0, 3, 1, 2)                # [B, 3, h, w]
    tgt = torch.from_numpy(np.ascontiguousarray(tgt)).float().to(device) / 127.5 - 1.0
    return ctx, act, tgt  # [B,K,3,h,w], [B,K,2], [B,3,h,w]


def _action_at(packed, episode, frame_idx) -> np.ndarray:
    """Action vector [actionX, hasClicked] at a frame index, clamped + NaN-safe."""
    e = packed.episodes[episode]
    idx = int(np.clip(frame_idx, 0, e["length"] - 1))
    lab = packed.labels[e["frame_offset"] + idx]
    ax = float(lab["actionX"])
    ahc = float(lab["actionHasClicked"])
    if np.isnan(ax):
        ax = 0.0
    if np.isnan(ahc) or ahc < 0:
        ahc = 0.0
    return np.array([ax, ahc], dtype=np.float32)


def _precompute_action_variants(packed, ep, start, K, seed):
    """true / earlier / later action arrays for all windows, plus a shuffled copy."""
    n = len(ep)
    true = np.zeros((n, K, 2), dtype=np.float32)
    earlier = np.zeros_like(true)
    later = np.zeros_like(true)
    for i, (e, s) in enumerate(zip(ep.tolist(), start.tolist())):
        for j in range(K):
            true[i, j] = _action_at(packed, e, s + j)
            earlier[i, j] = _action_at(packed, e, s + j - 1)   # one frame earlier
            later[i, j] = _action_at(packed, e, s + j + 1)     # one frame later
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    shuffled = true[perm]
    return {"true": true, "zeroed": np.zeros_like(true), "shuffled": shuffled,
            "earlier": earlier, "later": later}


def _pred_metrics(pred, tgt) -> dict:
    return {"l1": l1(pred, tgt), "mse": mse(pred, tgt),
            "psnr": psnr(pred, tgt), "ssim": ssim(pred, tgt)}


def _accum(acc, batch_metrics, weight):
    for k, v in batch_metrics.items():
        acc[k] = acc.get(k, 0.0) + v * weight
    return acc


def _finalize(acc, total):
    return {k: v / total for k, v in acc.items()}


def _ar_mask(packed, sidecar, ep, start, K):
    """Boolean mask: is each window action-relevant (target-step heading change)?"""
    out = []
    for e, s in zip(ep.tolist(), start.tolist()):
        d = sidecar.heading_change(e, s + K - 1)
        out.append(not np.isnan(d) and abs(d) >= _HEADING_EPS)
    return np.array(out, dtype=bool)


# --------------------------------------------------------------------------- #
# (a) one-step metrics  +  (b) action-use ablation
# --------------------------------------------------------------------------- #

def one_step_eval(packed, model, ep, start, K, sidecar, action_variants,
                  batch=256, device="cuda") -> dict:
    """One-step metrics for each action variant, overall and action-relevant.

    `action_variants` is a dict name -> array [N,K,2]. Persistence is computed
    once (it ignores the action). Returns counts and per-variant metric dicts.
    """
    n = len(ep)
    results = {"persistence": None, "variants": {}}
    pers_acc = {"overall": {}, "ar": {}}
    var_acc = {name: {"overall": {}, "ar": {}} for name in action_variants}
    n_total = 0
    n_ar = 0
    with torch.no_grad():
        for i in range(0, n, batch):
            ep_b = ep[i: i + batch]
            st_b = start[i: i + batch]
            ctx, act_true, tgt = _make_batch(packed, ep_b, st_b, K, device)
            ar = _ar_mask(packed, sidecar, ep_b, st_b, K)
            B = ctx.shape[0]
            n_total += B
            n_ar += int(ar.sum())

            pers = ctx[:, -1]
            pm = _pred_metrics(pers, tgt)
            _accum(pers_acc["overall"], pm, B)
            if ar.any():
                _accum(pers_acc["ar"], _pred_metrics(pers[ar], tgt[ar]), int(ar.sum()))

            for name, act_arr in action_variants.items():
                act = torch.from_numpy(np.ascontiguousarray(act_arr[i: i + batch])).to(device)
                pred = model.predict(ctx, act)
                vm = _pred_metrics(pred, tgt)
                _accum(var_acc[name]["overall"], vm, B)
                if ar.any():
                    _accum(var_acc[name]["ar"], _pred_metrics(pred[ar], tgt[ar]),
                           int(ar.sum()))

    results["persistence"] = {"overall": _finalize(pers_acc["overall"], n_total),
                              "action_relevant": _finalize(pers_acc["ar"], n_ar)}
    for name in action_variants:
        results["variants"][name] = {
            "overall": _finalize(var_acc[name]["overall"], n_total),
            "action_relevant": _finalize(var_acc[name]["ar"], n_ar),
        }
    results["n_total"] = n_total
    results["n_action_relevant"] = n_ar
    return results


# --------------------------------------------------------------------------- #
# (c) rollouts
# --------------------------------------------------------------------------- #

@torch.no_grad()
def rollout(packed, model, episode, start, K, steps, action_override=None,
            device="cuda") -> torch.Tensor:
    """Autoregressive rollout of `steps` frames from a real K-frame context.

    Returns the predicted frames [steps, 3, H, W] in [-1, 1]. With no override the
    TRUE action sequence is supplied; `action_override` (a [1,K,2] tensor) holds a
    constant action instead (for the counterfactual test).
    """
    e = packed.episodes[episode]
    off = e["frame_offset"]
    L = e["length"]
    acts = packed.action_vectors(packed.labels[off: off + L])
    acts = np.nan_to_num(acts, nan=0.0).astype(np.float32)

    ctx_u8 = packed.images[off + start: off + start + K]  # [K,H,W,3] uint8
    frames = frames_to_tensor(ctx_u8).to(device)          # [K,3,H,W] in [-1,1]
    preds = []
    for t in range(steps):
        c = frames[t: t + K].unsqueeze(0)
        if action_override is None:
            a = torch.from_numpy(acts[start + t: start + t + K]).to(device).unsqueeze(0)
        else:
            a = action_override
        nxt = model.predict(c, a)
        frames = torch.cat([frames, nxt], dim=0)
        preds.append(nxt[0])
    return torch.stack(preds, dim=0)


def _gt_after(packed, episode, start, K, steps, device):
    e = packed.episodes[episode]
    off = e["frame_offset"]
    gt = packed.images[off + start + K: off + start + K + steps]  # [steps,H,W,3] uint8
    return frames_to_tensor(gt).to(device)  # [steps,3,H,W] in [-1,1]


def _sample_clips(packed, split, K, steps, n_clips):
    ep, start = packed.enumerate_windows(split, K, 1, require_safe=True)
    keep = [(int(e), int(s)) for e, s in zip(ep.tolist(), start.tolist())
            if s + K + steps <= packed.episodes[int(e)]["length"]]
    if len(keep) <= n_clips:
        return keep
    idx = np.linspace(0, len(keep) - 1, n_clips).astype(int)
    return [keep[i] for i in idx]


def rollout_eval(packed, model, split, K, steps=40, n_clips=40, device="cuda") -> dict:
    """Error vs horizon (L1 in [0,1]) for the model and persistence, over clips."""
    clips = _sample_clips(packed, split, K, steps, n_clips)
    model_err = np.zeros(steps)
    pers_err = np.zeros(steps)
    with torch.no_grad():
        for ep, s in clips:
            preds = rollout(packed, model, ep, s, K, steps, device=device)
            gt = _gt_after(packed, ep, s, K, steps, device)
            err = (preds - gt).abs().mean(dim=(1, 2, 3)).cpu().numpy() / 2.0
            model_err += err
            # persistence repeats the last context frame; at step t that is frame s+K+t-1
            e = packed.episodes[ep]
            off = e["frame_offset"]
            prev = packed.images[off + s + K - 1: off + s + K - 1 + steps]
            curr = packed.images[off + s + K: off + s + K + steps]
            pers_err += np.abs(prev.astype(np.float32) - curr.astype(np.float32)).mean(
                axis=(1, 2, 3)) / 255.0
    n = len(clips)
    return {
        "n_clips": n,
        "horizon_model_l1": (model_err / n).tolist(),
        "horizon_persistence_l1": (pers_err / n).tolist(),
        "model_beats_persistence_until_step": _last_better(model_err, pers_err),
    }


def _last_better(model_err, pers_err):
    """Last 1-based horizon step at which the model's mean L1 <= persistence's."""
    last = -1
    for t in range(len(model_err)):
        if model_err[t] < pers_err[t]:
            last = t + 1
    return last


# --------------------------------------------------------------------------- #
# (d) action following
# --------------------------------------------------------------------------- #

def _torch_to_u8(t):
    """CHW float in [-1,1] -> HWC uint8 in [0,255] (for the shift estimator)."""
    t = t.float().detach().cpu()
    if t.ndim == 4:
        t = t.permute(0, 2, 3, 1)   # [N,3,H,W] -> [N,H,W,3]
    elif t.ndim == 3:
        t = t.permute(1, 2, 0)      # [3,H,W] -> [H,W,3]
    u = (t * 0.5 + 0.5) * 255.0
    return np.clip(u.numpy(), 0, 255).astype(np.uint8)


def action_following_eval(packed, model, split, K, sidecar, n_windows=2000,
                          device="cuda") -> dict:
    """On action-relevant windows, compare inter-frame horizontal image shift."""
    ep, start = packed.enumerate_windows(split, K, 1, require_safe=True)
    ar = _ar_mask(packed, sidecar, ep, start, K)
    idx = np.where(ar)[0]
    if len(idx) > n_windows:
        idx = np.sort(np.random.default_rng(0).choice(idx, size=n_windows, replace=False))

    real_shift, model_shift, zero_shift, heading = [], [], [], []
    with torch.no_grad():
        for i in idx.tolist():
            e, s = int(ep[i]), int(start[i])
            w = packed.window(e, s, K, 1)
            g = s + K - 1
            prev_u8 = w.frames[K - 1]           # last context frame
            next_u8 = w.frames[K]               # ground-truth next frame
            ctx, act, _ = _make_batch(packed, ep[i: i + 1], start[i: i + 1], K, device)
            pred = model.predict(ctx, act)[0]
            pred_zero = model.predict(ctx, torch.zeros_like(act))[0]

            real_shift.append(estimate_shift_x(prev_u8, next_u8))
            model_shift.append(estimate_shift_x(prev_u8, _torch_to_u8(pred)))
            zero_shift.append(estimate_shift_x(prev_u8, _torch_to_u8(pred_zero)))
            heading.append(sidecar.heading_change(e, g))

    real_shift = np.array(real_shift)
    model_shift = np.array(model_shift)
    zero_shift = np.array(zero_shift)
    heading = np.array(heading)
    noise = np.abs(real_shift) > 0.5  # pixels; windows where a real turn is visible

    return {
        "n_windows": len(real_shift),
        "mean_abs_shift_real_px": float(np.abs(real_shift).mean()),
        "mean_abs_shift_model_px": float(np.abs(model_shift).mean()),
        "mean_abs_shift_zeroed_px": float(np.abs(zero_shift).mean()),
        "corr_model_vs_real_shift": float(_pearson(model_shift, real_shift)),
        "corr_model_vs_heading": float(_pearson(model_shift, heading)),
        "corr_real_vs_heading": float(_pearson(real_shift, heading)),
        "sign_agreement_model_vs_real": float(
            (np.sign(model_shift) == np.sign(real_shift))[noise].mean()) if noise.any() else None,
        "sign_agreement_zeroed_vs_real": float(
            (np.sign(zero_shift) == np.sign(real_shift))[noise].mean()) if noise.any() else None,
    }


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom < 1e-12:
        return float("nan")
    return float((a * b).sum() / denom)


# --------------------------------------------------------------------------- #
# (e) counterfactual direction
# --------------------------------------------------------------------------- #

def counterfactual_eval(packed, model, split, K, sidecar, steps=15, max_contexts=50,
                        eps=0.05, device="cuda") -> dict:
    """Roll contexts with >=2 reachable doors on opposite sides under held clicks."""
    ep, start = packed.enumerate_windows(split, K, 1, require_safe=True)
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
        if len(contexts) >= max_contexts:
            break

    left_shift, right_shift, none_shift = [], [], []
    with torch.no_grad():
        for e, s, xl, xr in contexts:
            def held(x, clicked):
                a = np.full((1, K, 2), [x, 1.0 if clicked else 0.0], dtype=np.float32)
                return torch.from_numpy(a).to(device)
            pl = rollout(packed, model, e, s, K, steps, held(xl, True), device=device)
            pr = rollout(packed, model, e, s, K, steps, held(xr, True), device=device)
            pn = rollout(packed, model, e, s, K, steps, held(0.0, False), device=device)
            # net content shift between first and last predicted frame
            left_shift.append(estimate_shift_x(_torch_to_u8(pl[0]), _torch_to_u8(pl[-1])))
            right_shift.append(estimate_shift_x(_torch_to_u8(pr[0]), _torch_to_u8(pr[-1])))
            none_shift.append(estimate_shift_x(_torch_to_u8(pn[0]), _torch_to_u8(pn[-1])))

    left_shift = np.array(left_shift)
    right_shift = np.array(right_shift)
    none_shift = np.array(none_shift)
    n = len(left_shift)
    return {
        "n_contexts": n,
        "mean_shift_left_px": float(left_shift.mean()),
        "mean_shift_right_px": float(right_shift.mean()),
        "mean_shift_none_px": float(none_shift.mean()),
        # clicking left should turn the view left -> content moves right (+)
        "share_left_turns_left": float((left_shift > 0.5).mean()) if n else None,
        "share_right_turns_right": float((right_shift < -0.5).mean()) if n else None,
        # the model distinguishes the two directions
        "share_left_shift_gt_right_shift": float((left_shift > right_shift).mean()) if n else None,
        "share_both_correct_directions": float(
            ((left_shift > 0.5) & (right_shift < -0.5)).mean()) if n else None,
    }


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #

def run_harness(packed, model, split, K, sidecar, seed=0, batch=256, rollout_steps=40,
                rollout_clips=40, cf_steps=15, device="cuda") -> dict:
    ep, start = packed.enumerate_windows(split, K, 1, require_safe=True)
    action_variants = _precompute_action_variants(packed, ep, start, K, seed)

    results = {
        "split": split,
        "K": K,
        "action_relevant_definition":
            f"|target-step heading change| >= {ACTION_RELEVANT_DEG} deg",
        "one_step": one_step_eval(packed, model, ep, start, K, sidecar,
                                  action_variants, batch, device),
        "rollout": rollout_eval(packed, model, split, K, rollout_steps, rollout_clips,
                                device),
        "action_following": action_following_eval(packed, model, split, K, sidecar,
                                                  device=device),
        "counterfactual": counterfactual_eval(packed, model, split, K, sidecar,
                                              cf_steps, device=device),
    }
    return results


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate a trained next-frame model")
    ap.add_argument("--packed", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rollout-steps", dest="rollout_steps", type=int, default=40)
    ap.add_argument("--rollout-clips", dest="rollout_clips", type=int, default=40)
    ap.add_argument("--cf-steps", dest="cf_steps", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    packed = PackedDataset(args.packed, in_memory=True)
    model, payload = load_checkpoint(args.checkpoint, device)
    K = payload["model_config"]["K"]
    sidecar = load_analysis(packed)

    results = run_harness(packed, model, args.split, K, sidecar, batch=args.batch,
                          rollout_steps=args.rollout_steps,
                          rollout_clips=args.rollout_clips,
                          cf_steps=args.cf_steps, device=device)
    results["checkpoint"] = args.checkpoint
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    # compact console summary of the key numbers
    os_ = results["one_step"]
    print("=== one-step (overall / action-relevant) ===")
    print(f"  n windows: {os_['n_total']} (ar {os_['n_action_relevant']})")
    for name, v in os_["variants"].items():
        print(f"  {name:9s} L1 {v['overall']['l1']:.4f} / {v['action_relevant']['l1']:.4f}  "
              f"PSNR {v['overall']['psnr']:.2f}")
    print(f"  persistence L1 {os_['persistence']['overall']['l1']:.4f} / "
          f"{os_['persistence']['action_relevant']['l1']:.4f}")
    print("=== rollout ===")
    print(f"  model beats persistence until step {results['rollout']['model_beats_persistence_until_step']}")
    print("=== action following ===")
    for k, v in results["action_following"].items():
        print(f"  {k}: {v}")
    print("=== counterfactual ===")
    for k, v in results["counterfactual"].items():
        print(f"  {k}: {v}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
