"""P2 deployment-realistic conditioning: hold-last future actions.

At play time only the held action a_{t-1} is known — it persists until the next
click. So the future-H model should be fed, for its lookahead, the **last context
action repeated H times**, not the (unavailable) true future actions that
``train_future.py`` uses. Onset windows are "a click occurs at the final context
step" (actionX changes at a_{K-1}) and a turn develops in the horizon — exactly
what the model sees at the moment it must act.

After training, the script runs the hold-last ablation (vary only the held click
a_{K-1}: true / zeroed / shuffled) and the P1 click counterfactual via
``probe_click``.

Run: .venv/bin/python train_hold.py --name hold10 --packed data/pilot_v1_64x48 \
       --horizon 10 --mirror --max-steps 6000
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
from train_future import _build, _ONSET_AHEAD, _onset_mask


def _onset(packed, sidecar, ep, start, K, H, kind):
    if kind == "quiet":
        return _onset_mask(sidecar, ep, start, K, H)
    return _click_onset(packed, sidecar, ep, start, K, H)


def _click_onset(packed, sidecar, ep, start, K, H):
    """Onset = actionX changes at the final context step AND a turn develops."""
    out = np.zeros(len(ep), dtype=bool)
    for i, (e, s) in enumerate(zip(ep.tolist(), start.tolist())):
        g = s + K - 1
        off = packed.episodes[int(e)]["frame_offset"]
        a = packed.action_vectors(packed.labels[off + s: off + s + K + H])
        ax0, ax1 = a[K - 2, 0], a[K - 1, 0]
        if np.isnan(ax0) or np.isnan(ax1) or abs(ax1 - ax0) < 1e-4:
            continue
        ahead = max((abs(sidecar.heading_change(e, g + j)) for j in range(1, H + 1)),
                    default=0.0)
        if ahead >= _ONSET_AHEAD:
            out[i] = True
    return out


def _make_batch_hold(packed, ep, start, K, H, device):
    ctx, act, tgt = [], [], []
    for e, s in zip(ep.tolist(), start.tolist()):
        e2 = packed.episodes[int(e)]
        off = e2["frame_offset"]
        fr = packed.images[off + s: off + s + K + H]
        ctx.append(fr[:K].transpose(0, 3, 1, 2))
        tgt.append(fr[K + H - 1].transpose(2, 0, 1))
        a = packed.action_vectors(packed.labels[off + s: off + s + K + H])
        a = np.nan_to_num(a, nan=0.0).astype(np.float32)
        a[K:] = a[K - 1]                 # hold-last: repeat the last context action
        act.append(a)
    ctx = torch.from_numpy(np.ascontiguousarray(np.stack(ctx))).float().to(device) / 127.5 - 1.0
    tgt = torch.from_numpy(np.ascontiguousarray(np.stack(tgt))).float().to(device) / 127.5 - 1.0
    act = torch.from_numpy(np.stack(act)).float().to(device)
    return ctx, act, tgt


def _sample_batch(ep, start, onset, B, rng):
    on = np.where(onset)[0]
    off = np.where(~onset)[0]
    n_on = min(len(on), B // 2)
    n_off = B - n_on
    pick = list(rng.choice(on, n_on, replace=False)) + list(
        rng.choice(off, n_off, replace=(n_off > len(off))))
    return ep[pick], start[pick]


def _val_l1(packed, model, ep, start, K, H, batch, device):
    tot, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ep), batch):
            ctx, act, tgt = _make_batch_hold(packed, ep[i:i + batch], start[i:i + batch],
                                             K, H, device)
            pred = model.predict(ctx, act)
            tot += F.l1_loss(pred, tgt).item() * len(ep[i:i + batch])
            n += len(ep[i:i + batch])
    model.train()
    return tot / max(1, n)


def train(config):
    _set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = config["mixed_precision"] == "bf16" and device == "cuda"
    K, H = config["K"], config["horizon"]

    packed = PackedDataset(config["packed"], in_memory=True)
    ep, start = packed.enumerate_windows("train", K, H, require_safe=True)
    ep = ep.astype(np.int64)
    start = start.astype(np.int64)
    sidecar = load_analysis(packed)
    onset = _onset(packed, sidecar, ep, start, K, H, config.get("onset", "click"))
    print(f"train windows {len(ep)}, onset({config.get('onset','click')}) "
          f"{int(onset.sum())} ({onset.sum() / len(ep) * 100:.1f}%)", flush=True)

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
    full_cfg["hold_last"] = True
    full_cfg["onset"] = config.get("onset", "click")
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**full_cfg, "n_train_windows": len(ep),
                   "n_onset_windows": int(onset.sum()), "n_params": n_params,
                   "device": device}, f, indent=1)

    best_val = float("inf")
    running = 0.0
    t0 = time.time()
    rng = np.random.default_rng(config["seed"] + 99)
    B = config["batch_size"]
    for step in range(1, config["max_steps"] + 1):
        eb, sb = _sample_batch(ep, start, onset, B, rng)
        ctx, act, tgt = _make_batch_hold(packed, eb, sb, K, H, device)
        if config.get("mirror"):
            m = torch.from_numpy(rng.random(B) < 0.5).to(device)
            idx = m.nonzero(as_tuple=False).squeeze(1)
            if len(idx):
                ctx[idx] = torch.flip(ctx[idx], dims=[-1])
                tgt[idx] = torch.flip(tgt[idx], dims=[-1])
                act[idx, :, 0] = -act[idx, :, 0]
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
            val_loss = _val_l1(packed, model, val_ep, val_start, K, H, B, device)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(os.path.join(out_dir, "best.pt"), model, full_cfg,
                                val_loss, step)
            print(f"step {step} train {train_loss:.5f} val {val_loss:.5f} "
                  f"best {best_val:.5f} ({time.time() - t0:.0f}s)", flush=True)

    save_checkpoint(os.path.join(out_dir, "last.pt"), model, full_cfg,
                    float(best_val), config["max_steps"])
    print(json.dumps({"name": config["name"], "n_params": n_params,
                      "best_val_loss": best_val, "seconds": time.time() - t0,
                      "n_onset_windows": int(onset.sum())}, indent=1), flush=True)
    return out_dir


def ablate_hold(packed, model, sidecar, K, H, split, max_windows, onset_kind="click"):
    device = next(model.parameters()).device
    ep, start = packed.enumerate_windows(split, K, H, require_safe=True)
    ep = ep.astype(np.int64)
    start = start.astype(np.int64)
    onset = _onset(packed, sidecar, ep, start, K, H, onset_kind)
    ep, start = ep[onset], start[onset]
    if len(ep) > max_windows:
        idx = np.sort(np.random.default_rng(0).choice(len(ep), max_windows, replace=False))
        ep, start = ep[idx], start[idx]
    print(f"onset({onset_kind}) windows: {len(ep)}", flush=True)

    # per-window context actions + held click
    seqs = []
    for e, s in zip(ep.tolist(), start.tolist()):
        off = packed.episodes[int(e)]["frame_offset"]
        a = packed.action_vectors(packed.labels[off + s: off + s + K + H])
        a = np.nan_to_num(a, nan=0.0).astype(np.float32)
        seqs.append(a[:K].copy())          # [K,2] context actions
    seqs = np.stack(seqs)                  # [N,K,2]
    last = seqs[:, K - 1:K].copy()         # [N,1,2] the held click
    zero = np.zeros_like(last)
    shuffled = last[np.random.default_rng(0).permutation(len(ep))]

    variants = {"true": last, "zeroed": zero, "shuffled": shuffled}
    acc = {n: 0.0 for n in variants}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ep), 64):
            eb, sb = ep[i:i + 64], start[i:i + 64]
            ctx, _, tgt = _make_batch_hold(packed, eb, sb, K, H, device)
            base = seqs[i:i + 64, :K - 1]              # context except the last action
            for name, lv in variants.items():
                ctx_act = np.concatenate([base, lv[i:i + 64]], axis=1)  # [B,K,2]
                fut = np.repeat(lv[i:i + 64], H, axis=1)                # [B,H,2]
                full = np.concatenate([ctx_act, fut], axis=1)           # [B,K+H,2]
                act = torch.from_numpy(full).float().to(device)
                pred = model.predict(ctx, act)
                acc[name] += F.l1_loss(pred, tgt).item() * len(eb)

    n = len(ep)
    print(f"{'held click':14s} H-step L1 (n={n})")
    for name in variants:
        print(f"  {name:14s} {acc[name] / n:.5f}")
    return {name: acc[name] / n for name in variants}


def main(argv=None):
    from train import DEFAULT_CONFIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hold10")
    ap.add_argument("--packed", default="data/pilot_v1_64x48")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--onset", choices=("click", "quiet"), default="click")
    ap.add_argument("--mirror", action="store_true")
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
    else:
        config = dict(DEFAULT_CONFIG)
        config.update({k: v for k, v in vars(args).items() if v is not None})
        out_dir = train(config)
        model, _ = load_checkpoint(os.path.join(out_dir, "best.pt"), "cuda")
        K, H = args.K, args.horizon

    ablate_hold(packed, model, sidecar, K, H, "val", args.max_windows,
                onset_kind=(payload.get("config", {}).get("onset", "click")
                            if args.checkpoint else args.onset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
