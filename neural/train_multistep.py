"""Multi-step (autoregressive) training for the baseline next-frame model.

One-step L1 gives the model no reason to read the action: at 20 Hz the action's
effect on the *next* frame is ~0.03 px (see onset_effect.py). So this script
trains with a rollout objective: over H steps it feeds the model its own
(detached) predictions and the true actions, and sums the L1 over the horizon,
so ignoring the action is penalized where the turn actually accumulates. It also
oversamples turn-onset windows (context shows no turn yet, target horizon does),
which is exactly where the action is the only source of the turn.

Run: .venv/bin/python train_multistep.py --name ms8 --packed data/pilot_v1_64x48 \
       --horizon 8 --onset-frac 0.5 --max-steps 2500
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
from model import build_model, count_parameters, save_checkpoint
from train import _set_seed, evaluate_val

_DEG = np.deg2rad(1.0)
_ONSET_BEFORE = np.deg2rad(0.2)   # context must NOT be turning yet
_ONSET_AHEAD = np.deg2rad(0.5)    # but a turn must appear in the horizon


def _onset_mask(packed, sidecar, ep, start, K, H):
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


def _make_ms_batch(packed, ep, start, K, H, device):
    ctx, acts, tgt = [], [], []
    for e, s in zip(ep.tolist(), start.tolist()):
        e2 = packed.episodes[int(e)]
        off = e2["frame_offset"]
        fr = packed.images[off + s: off + s + K + H]          # [K+H, h, w, 3]
        ctx.append(fr[:K].transpose(0, 3, 1, 2))              # [K,3,h,w]
        tgt.append(fr[K:].transpose(0, 3, 1, 2))              # [H,3,h,w]
        lab = packed.labels[off + s: off + s + K + H]
        a = packed.action_vectors(lab)                        # [K+H, 2]
        acts.append(np.nan_to_num(a, nan=0.0).astype(np.float32))
    ctx = torch.from_numpy(np.ascontiguousarray(np.stack(ctx))).float().to(device) / 127.5 - 1.0
    tgt = torch.from_numpy(np.ascontiguousarray(np.stack(tgt))).float().to(device) / 127.5 - 1.0
    acts = torch.from_numpy(np.stack(acts)).float().to(device)
    return ctx, acts, tgt


def _sample_batch(ep, start, onset, B, rng):
    on = np.where(onset)[0]
    off = np.where(~onset)[0]
    n_on = min(len(on), B // 2)
    n_off = B - n_on
    pick = list(rng.choice(on, n_on, replace=False)) + list(
        rng.choice(off, n_off, replace=(n_off > len(off))))
    return ep[pick], start[pick]


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
    onset = _onset_mask(packed, sidecar, ep, start, K, H)
    n_on = int(onset.sum())
    print(f"train windows {len(ep)}, onset windows {n_on} "
          f"({n_on / max(1, len(ep)) * 100:.1f}%)", flush=True)

    # val: one-step teacher-forced (a stable proxy for checkpoint selection)
    val_ep, val_start = packed.enumerate_windows("val", K, 1, require_safe=True)
    if len(val_ep) > config["val_windows"]:
        rngv = np.random.default_rng(config["seed"] + 1)
        vi = np.sort(rngv.permutation(len(val_ep))[: config["val_windows"]])
        val_ep, val_start = val_ep[vi], val_start[vi]

    model = build_model(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config["weight_decay"])
    n_params = count_parameters(model)

    out_dir = os.path.join(config["out"], config["name"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**config, "n_train_windows": len(ep), "n_onset_windows": n_on,
                   "n_params": n_params, "device": device}, f, indent=1)

    log = []
    best_val = float("inf")
    running = 0.0
    t0 = time.time()
    rng = np.random.default_rng(config["seed"] + 99)
    B = config["batch_size"]
    for step in range(1, config["max_steps"] + 1):
        eb, sb = _sample_batch(ep, start, onset, B, rng)
        ctx, acts, tgt = _make_ms_batch(packed, eb, sb, K, H, device)

        opt.zero_grad(set_to_none=True)
        loss = 0.0
        cur = ctx
        for t in range(H):
            with torch.autocast(device_type=device, dtype=torch.bfloat16,
                                enabled=use_bf16):
                pred = model.predict(cur, acts[:, t: t + K])
            loss = loss + F.l1_loss(pred.float(), tgt[:, t])
            nxt = pred if config.get("no_detach") else pred.detach()
            cur = torch.cat([cur[:, 1:], nxt.unsqueeze(1)], dim=1)
        loss = loss / H
        if config.get("grad_clip") is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        loss.backward()
        opt.step()
        running += float(loss.item())

        if step % config["val_every"] == 0 or step == config["max_steps"]:
            train_loss = running / config["val_every"]
            running = 0.0
            val_loss = evaluate_val(model, packed, val_ep, val_start, K, 1,
                                    config["batch_size"], device, use_bf16)
            log.append({"step": step, "train_loss": train_loss,
                        "val_loss": val_loss, "elapsed_s": time.time() - t0})
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(os.path.join(out_dir, "best.pt"), model, config,
                                val_loss, step)
            print(f"step {step} train {train_loss:.5f} val {val_loss:.5f} "
                  f"best {best_val:.5f} ({time.time() - t0:.0f}s)", flush=True)

    save_checkpoint(os.path.join(out_dir, "last.pt"), model, config,
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


def main(argv=None) -> int:
    from train import DEFAULT_CONFIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ms8")
    ap.add_argument("--packed", default="data/pilot_v1_64x48")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--onset-frac", dest="onset_frac", type=float, default=0.5)
    ap.add_argument("--batch", dest="batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", dest="val_every", type=int, default=250)
    ap.add_argument("--val-windows", dest="val_windows", type=int, default=1024)
    ap.add_argument("--conditioning", default="channels")
    ap.add_argument("--no-detach", dest="no_detach", action="store_true",
                    help="backprop through the whole rollout (cross-step credit assignment)")
    ap.add_argument("--grad-clip", dest="grad_clip", type=float, default=None)
    ap.add_argument("--train-fraction", dest="train_fraction", type=float, default=None)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args(argv)

    config = dict(DEFAULT_CONFIG)
    config.update({k: v for k, v in vars(args).items() if v is not None})
    config["H"] = 1  # keep evaluate_val's window H=1; the multi-step H is "horizon"
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
