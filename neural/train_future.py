"""H-step teacher-forced lookahead: make the action usable by giving it a horizon.

The one-step model cannot use the action because a click moves the *next* frame by
~0.03 px (see onset_effect.py). The autoregressive multi-step attempts (ms8, ms10_btp)
failed too, because the model's own compounding rollout error dominated the turn
signal. This script removes both problems at once: the model is trained to predict
the frame **H steps ahead**, in one forward pass, given the K context frames and the
**future action sequence** (K+H actions). The target frame contains the accumulated
turn (~10 px at H=10), and only the lookahead actions — not the context — say which
way. So the pixel loss now *directly* rewards reading the action.

Conditioning is FiLM (the variable-length action sequence is flattened to a vector),
with ``n_action_frames = K + H``.

Training is a single forward per step (fast). After training, the script runs the
decision-relevant ablation on turn-onset val windows at the H-step horizon:

    true vs zeroed vs shuffled *future* actions  ->  H-step L1

If the model is genuinely action-responsive, true actions should roll out clearly
better than zeroed/shuffled (by roughly the accumulated-turn cost).

Run: .venv/bin/python train_future.py --name fut10 --packed data/pilot_v1_64x48 \
       --horizon 10 --onset-frac 0.5 --max-steps 6000
     .venv/bin/python train_future.py --checkpoint runs/fut10/best.pt --packed ... \
       --ablate-only
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from analysis import load_analysis
from dataset import PackedDataset
from model import build_model, count_parameters, save_checkpoint, load_checkpoint
from train import _set_seed

_ONSET_BEFORE = np.deg2rad(0.2)   # context must NOT be turning yet
_ONSET_AHEAD = np.deg2rad(0.5)    # but a turn must appear in the lookahead


def _onset_mask(sidecar, ep, start, K, H):
    out = np.zeros(len(ep), dtype=bool)
    for i, (e, s) in enumerate(zip(ep.tolist(), start.tolist())):
        g = s + K - 1
        b = sidecar.heading_change(e, g)
        if np.isnan(b) or abs(b) >= _ONSET_BEFORE:
            continue
        ahead = max((abs(sidecar.heading_change(e, g + j)) for j in range(1, H + 1)),
                    default=0.0)
        if ahead >= _ONSET_AHEAD:
            out[i] = True
    return out


def _sample_batch(ep, start, onset, B, rng):
    on = np.where(onset)[0]
    off = np.where(~onset)[0]
    n_on = min(len(on), B // 2)
    n_off = B - n_on
    pick = list(rng.choice(on, n_on, replace=False)) + list(
        rng.choice(off, n_off, replace=(n_off > len(off))))
    return ep[pick], start[pick]


def _make_batch(packed, ep, start, K, H, device):
    ctx, act, tgt = [], [], []
    for e, s in zip(ep.tolist(), start.tolist()):
        e2 = packed.episodes[int(e)]
        off = e2["frame_offset"]
        fr = packed.images[off + s: off + s + K + H]      # [K+H, h, w, 3]
        ctx.append(fr[:K].transpose(0, 3, 1, 2))          # [K, 3, h, w]
        tgt.append(fr[K + H - 1].transpose(2, 0, 1))      # [3, h, w]  frame s+K+H-1
        lab = packed.labels[off + s: off + s + K + H]
        a = packed.action_vectors(lab)                    # [K+H, 2]
        act.append(np.nan_to_num(a, nan=0.0).astype(np.float32))
    ctx = torch.from_numpy(np.ascontiguousarray(np.stack(ctx))).float().to(device) / 127.5 - 1.0
    tgt = torch.from_numpy(np.ascontiguousarray(np.stack(tgt))).float().to(device) / 127.5 - 1.0
    act = torch.from_numpy(np.stack(act)).float().to(device)
    return ctx, act, tgt


def _hstep_l1(packed, model, ep, start, K, H, batch, device):
    """Teacher-forced H-step val L1."""
    tot = 0.0
    n = 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ep), batch):
            eb, sb = ep[i:i + batch], start[i:i + batch]
            ctx, act, tgt = _make_batch(packed, eb, sb, K, H, device)
            pred = model.predict(ctx, act)
            tot += F.l1_loss(pred, tgt).item() * len(eb)
            n += len(eb)
    model.train()
    return tot / max(1, n)


def _build(cfg):
    return build_model(dict(K=cfg["K"], base=cfg.get("base", 32), conditioning="film",
                            n_action_frames=cfg["K"] + cfg["horizon"]))


def train(config: dict) -> dict:
    _set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = config["mixed_precision"] == "bf16" and device == "cuda"
    K, H = config["K"], config["horizon"]

    packed = PackedDataset(config["packed"], in_memory=True)
    ep, start = packed.enumerate_windows("train", K, H, require_safe=True)
    ep = ep.astype(np.int64)
    start = start.astype(np.int64)
    if config.get("train_fraction") is not None:
        n = len(ep)
        rng0 = np.random.default_rng(config["seed"])
        idx = np.sort(rng0.permutation(n)[: max(1, int(round(n * config["train_fraction"])))])
        ep, start = ep[idx], start[idx]

    sidecar = load_analysis(packed)
    onset = _onset_mask(sidecar, ep, start, K, H)
    n_on = int(onset.sum())
    print(f"train windows {len(ep)}, onset windows {n_on} "
          f"({n_on / max(1, len(ep)) * 100:.1f}%)", flush=True)

    val_ep, val_start = packed.enumerate_windows("val", K, H, require_safe=True)
    val_ep = val_ep.astype(np.int64)
    val_start = val_start.astype(np.int64)
    if len(val_ep) > config["val_windows"]:
        rngv = np.random.default_rng(config["seed"] + 1)
        vi = np.sort(rngv.permutation(len(val_ep))[: config["val_windows"]])
        val_ep, val_start = val_ep[vi], val_start[vi]

    model = _build(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config["weight_decay"])
    n_params = count_parameters(model)

    out_dir = os.path.join(config["out"], config["name"])
    os.makedirs(out_dir, exist_ok=True)
    full_cfg = dict(config)
    full_cfg["n_action_frames"] = K + H
    full_cfg["conditioning"] = "film"
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**full_cfg, "n_train_windows": len(ep), "n_onset_windows": n_on,
                   "n_params": n_params, "device": device}, f, indent=1)

    log = []
    best_val = float("inf")
    running = 0.0
    t0 = time.time()
    rng = np.random.default_rng(config["seed"] + 99)
    B = config["batch_size"]
    for step in range(1, config["max_steps"] + 1):
        eb, sb = _sample_batch(ep, start, onset, B, rng)
        ctx, act, tgt = _make_batch(packed, eb, sb, K, H, device)
        if config.get("mirror"):
            m = torch.from_numpy(rng.random(B) < 0.5).to(device)
            idx = m.nonzero(as_tuple=False).squeeze(1)
            if len(idx):
                ctx[idx] = torch.flip(ctx[idx], dims=[-1])
                tgt[idx] = torch.flip(tgt[idx], dims=[-1])
                act[idx, :, 0] = -act[idx, :, 0]   # mirror the screen X position
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            pred = model.predict(ctx, act)
        loss = F.l1_loss(pred.float(), tgt)
        loss.backward()
        opt.step()
        running += float(loss.item())

        if step % config["val_every"] == 0 or step == config["max_steps"]:
            train_loss = running / config["val_every"]
            running = 0.0
            val_loss = _hstep_l1(packed, model, val_ep, val_start, K, H,
                                 config["batch_size"], device)
            log.append({"step": step, "train_loss": train_loss,
                        "val_loss": val_loss, "elapsed_s": time.time() - t0})
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(os.path.join(out_dir, "best.pt"), model, full_cfg,
                                val_loss, step)
            print(f"step {step} train {train_loss:.5f} val {val_loss:.5f} "
                  f"best {best_val:.5f} ({time.time() - t0:.0f}s)", flush=True)

    save_checkpoint(os.path.join(out_dir, "last.pt"), model, full_cfg,
                    float(log[-1]["val_loss"] if log else 0.0), config["max_steps"])
    dt = time.time() - t0
    result = {"name": config["name"], "n_params": n_params, "steps": config["max_steps"],
              "best_val_loss": best_val, "seconds": dt,
              "steps_per_second": config["max_steps"] / dt if dt > 0 else 0.0,
              "n_train_windows": len(ep), "n_onset_windows": n_on}
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1), flush=True)
    return result


def ablate(packed, model, sidecar, K, H, split, max_windows):
    device = next(model.parameters()).device
    ep, start = packed.enumerate_windows(split, K, H, require_safe=True)
    ep = ep.astype(np.int64)
    start = start.astype(np.int64)
    onset = _onset_mask(sidecar, ep, start, K, H)
    ep, start = ep[onset], start[onset]
    if len(ep) > max_windows:
        idx = np.sort(np.random.default_rng(0).choice(len(ep), max_windows, replace=False))
        ep, start = ep[idx], start[idx]
    print(f"onset windows: {len(ep)}", flush=True)

    # full action sequences [N, K+H, 2]; future = last H frames of the sequence
    seqs = []
    for e, s in zip(ep.tolist(), start.tolist()):
        e2 = packed.episodes[int(e)]
        off = e2["frame_offset"]
        lab = packed.labels[off + s: off + s + K + H]
        a = packed.action_vectors(lab)
        seqs.append(np.nan_to_num(a, nan=0.0).astype(np.float32))
    seqs = np.stack(seqs)                       # [N, K+H, 2]
    future_true = seqs[:, K:].copy()            # [N, H, 2]
    zeroed = np.zeros_like(future_true)
    shuffled = future_true[np.random.default_rng(0).permutation(len(ep))]

    variants = {"true": future_true, "zeroed": zeroed, "shuffled": shuffled}
    acc = {name: 0.0 for name in variants}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ep), 64):
            eb, sb = ep[i:i + 64], start[i:i + 64]
            ctx, _, tgt = _make_batch(packed, eb, sb, K, H, device)
            ctx_act = seqs[i:i + 64, :K]                       # context actions fixed
            for name, fut in variants.items():
                full = np.concatenate([ctx_act, fut[i:i + 64]], axis=1)
                act = torch.from_numpy(full).float().to(device)
                pred = model.predict(ctx, act)
                acc[name] += F.l1_loss(pred, tgt).item() * len(eb)

    n = len(ep)
    print(f"{'future-actions':16s} H-step L1 (n={n})")
    for name in variants:
        print(f"  {name:14s} {acc[name] / n:.5f}")
    return {name: acc[name] / n for name in variants}


def counterfactual(packed, model, sidecar, K, H, max_contexts=40, eps=0.05):
    """H-step counterfactual direction: does the lookahead click steer the view?"""
    device = next(model.parameters()).device
    from metrics import estimate_shift_x
    ep, start = packed.enumerate_windows("val", K, H, require_safe=True)
    contexts = []
    for e, s in zip(ep.tolist(), start.tolist()):
        g = s + K - 1
        cands = [c for c in sidecar.candidates_at(e, g) if c["reachable"]]
        xs = [c["screenX"] for c in cands]
        left = [x for x in xs if x < -eps]
        right = [x for x in xs if x > eps]
        if left and right:
            contexts.append((e, s, min(left), max(right)))
        if len(contexts) >= max_contexts:
            break

    left_ok = right_ok = both = 0
    model.eval()
    with torch.no_grad():
        for e, s, xl, xr in contexts:
            e2 = packed.episodes[int(e)]
            off = e2["frame_offset"]
            fr = packed.images[off + s: off + s + K]
            ctx = torch.from_numpy(np.ascontiguousarray(fr.transpose(0, 3, 1, 2)[None])) \
                .float().to(device) / 127.5 - 1.0
            last_u8 = fr[K - 1]
            shifts = {}
            for tag, x, clicked in (("left", xl, 1.0), ("right", xr, 1.0), ("none", 0.0, 0.0)):
                fut = np.full((H, 2), [x, clicked], dtype=np.float32)
                lab = packed.labels[off + s: off + s + K]
                ctx_act = np.nan_to_num(packed.action_vectors(lab), nan=0.0).astype(np.float32)
                full = np.concatenate([ctx_act[None], fut[None]], axis=1)
                act = torch.from_numpy(full).float().to(device)
                pred = model.predict(ctx, act)[0]                 # [3,h,w]
                pred_u8 = pred.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5
                pred_u8 = (pred_u8 * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
                shifts[tag] = estimate_shift_x(last_u8, pred_u8)  # + = content right
            # + shift = content moved right = view turned LEFT toward a left door
            left_ok += int(shifts["left"] > shifts["none"])       # left click -> content right
            right_ok += int(shifts["right"] < shifts["none"])     # right click -> content left
            both += int(shifts["left"] > shifts["none"] > shifts["right"])
    n = len(contexts)
    print(f"counterfactual (H={H}, n={n}): left-correct {left_ok / max(1, n):.2f}, "
          f"right-correct {right_ok / max(1, n):.2f}, both {both / max(1, n):.2f}")
    return {"n": n, "left_correct": left_ok / max(1, n),
            "right_correct": right_ok / max(1, n), "both_correct": both / max(1, n)}


def main(argv=None) -> int:
    from train import DEFAULT_CONFIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fut10")
    ap.add_argument("--packed", default="data/pilot_v1_64x48")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--onset-frac", dest="onset_frac", type=float, default=0.5)
    ap.add_argument("--mirror", action="store_true",
                    help="mirror augmentation: flip frames horizontally + negate actionX")
    ap.add_argument("--batch", dest="batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", dest="val_every", type=int, default=250)
    ap.add_argument("--val-windows", dest="val_windows", type=int, default=1024)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--ablate-only", dest="ablate_only", action="store_true")
    ap.add_argument("--max-windows", dest="max_windows", type=int, default=300)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args(argv)

    packed = PackedDataset(args.packed, in_memory=True)
    sidecar = load_analysis(packed)

    if args.checkpoint:
        model, payload = load_checkpoint(args.checkpoint, "cuda")
        K = payload["model_config"]["K"]
        H = (payload["model_config"].get("n_action_frames") or (K + args.horizon)) - K
        print(f"loaded {args.checkpoint}: K={K} H={H}", flush=True)
    else:
        K, H = args.K, args.horizon
        config = dict(DEFAULT_CONFIG)
        config.update({k: v for k, v in vars(args).items() if v is not None})
        train(config)
        model, _ = load_checkpoint(os.path.join(args.out, args.name, "best.pt"), "cuda")

    ablate(packed, model, sidecar, K, H, "val", args.max_windows)
    counterfactual(packed, model, sidecar, K, H)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
