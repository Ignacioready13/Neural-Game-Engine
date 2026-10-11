"""Config-driven training for the baseline next-frame model.

Usage (from /neural):

    .venv/bin/python train.py --name baseline_64 --packed data/pilot_v1_64x48 \
        --max-steps 6000 --batch 64

Everything that matters is a config key; the full config is saved next to the
checkpoint so a run is reproducible and later models can reuse the same script.
Artifacts land in ``runs/<name>/`` (gitignored): ``config.json``, ``best.pt``
(lowest val loss), ``last.pt``, ``train_log.json`` (step/train/val losses) and
``result.json`` (a one-line summary incl. throughput).

Useful knobs for the stage's acceptance checks:
  --limit N            train on only the first N windows (the sanity overfit)
  --train-fraction F   train on a seeded random F fraction of windows (learning curve)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PackedDataset, WindowDataset
from model import build_model, count_parameters, save_checkpoint

DEFAULT_CONFIG = {
    "name": "baseline",
    "packed": "data/pilot_v1_64x48",
    "K": 4,
    "H": 1,
    "base": 32,
    "conditioning": "channels",  # "channels" (broadcast) or "film" (FiLM at scales)
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "max_steps": 6000,
    "seed": 0,
    "mixed_precision": "bf16",
    "loss": "l1",
    "val_every": 500,
    "val_windows": 1024,
    "num_workers": 2,
    "limit": None,           # int: cap the number of train windows (sanity overfit)
    "train_fraction": None,  # float in (0, 1]: subsample train windows (learning curve)
    "out": "runs",
}


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _subsample(win_ep, win_start, fraction, seed):
    n = len(win_ep)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.permutation(n)[: max(1, int(round(n * fraction)))])
    return win_ep[idx], win_start[idx]


def _select_train_windows(packed, config):
    ds = WindowDataset(packed, "train", K=config["K"], H=config["H"], normalize=True)
    win_ep, win_start = ds.win_episode, ds.win_start
    if config.get("limit") is not None:
        n = int(config["limit"])
        win_ep, win_start = win_ep[:n], win_start[:n]
    elif config.get("train_fraction") is not None:
        win_ep, win_start = _subsample(
            win_ep, win_start, float(config["train_fraction"]), config["seed"]
        )
    ds.win_episode = win_ep
    ds.win_start = win_start
    return ds, packed


def _select_val_windows(packed, config):
    ds = WindowDataset(packed, "val", K=config["K"], H=config["H"], normalize=True)
    win_ep, win_start = ds.win_episode, ds.win_start
    if config["val_windows"] and len(win_ep) > config["val_windows"]:
        rng = np.random.default_rng(config["seed"] + 1)
        idx = np.sort(rng.permutation(len(win_ep))[: config["val_windows"]])
        win_ep, win_start = win_ep[idx], win_start[idx]
    return win_ep, win_start


def _make_batch(packed, ep_arr, start_arr, K, H, device):
    """Materialize a batch of windows from (episode, start) arrays, on `device`."""
    ctx, act, tgt = [], [], []
    for e, s in zip(ep_arr.tolist(), start_arr.tolist()):
        w = packed.window(int(e), int(s), K, H)
        ctx.append(np.ascontiguousarray(w.context))          # [K, h, w, 3]
        act.append(np.ascontiguousarray(w.context_action))   # [K, 2]
        tgt.append(np.ascontiguousarray(w.target[0]))        # [h, w, 3]
    ctx = np.stack(ctx).transpose(0, 1, 4, 2, 3)             # [B, K, 3, h, w]
    ctx = torch.from_numpy(np.ascontiguousarray(ctx)).float().to(device) / 127.5 - 1.0
    act = torch.from_numpy(np.stack(act)).float().to(device)
    tgt = np.stack(tgt).transpose(0, 3, 1, 2)                # [B, 3, h, w]
    tgt = torch.from_numpy(np.ascontiguousarray(tgt)).float().to(device) / 127.5 - 1.0
    return ctx, act, tgt  # [B,K,3,h,w], [B,K,2], [B,3,h,w]


def evaluate_val(model, packed, val_ep, val_start, K, H, batch, device, use_bf16):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(val_ep), batch):
            ep = val_ep[i: i + batch]
            st = val_start[i: i + batch]
            ctx, act, tgt = _make_batch(packed, ep, st, K, H, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                out = model.predict(ctx, act)
            total += F.l1_loss(out.float(), tgt).item() * ctx.shape[0]
            n += ctx.shape[0]
    model.train()
    return total / n


def train(config: dict) -> dict:
    _set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = config["mixed_precision"] == "bf16" and device == "cuda"

    packed = PackedDataset(config["packed"], in_memory=True)
    train_ds, _ = _select_train_windows(packed, config)
    # num_workers>0 + pin_memory keeps the main process off the numpy conversion
    # path; with num_workers=0 the per-batch CPU work stalls the GPU loop (~6 vs
    # ~54 steps/s). persistent_workers avoids re-forking the (COW-shared) dataset.
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=(device == "cuda"),
        drop_last=False, persistent_workers=(config["num_workers"] > 0),
    )
    val_ep, val_start = _select_val_windows(packed, config)

    model = build_model(config).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config["weight_decay"])
    n_params = count_parameters(model)

    out_dir = os.path.join(config["out"], config["name"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**config, "n_train_windows": len(train_ds),
                   "n_params": n_params, "device": device}, f, indent=1)

    log = []
    best_val = float("inf")
    step = 0
    epoch = 0
    t0 = time.time()
    running = 0.0
    data_iter = iter(train_loader)
    while step < config["max_steps"]:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(train_loader)
            batch = next(data_iter)

        ctx = batch["context"].to(device, non_blocking=True)
        act = batch["context_action"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True).squeeze(1)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            out = model.predict(ctx, act)
        loss = F.l1_loss(out.float(), tgt)
        loss.backward()
        opt.step()

        running += float(loss.item())
        step += 1

        if step % config["val_every"] == 0 or step == config["max_steps"]:
            train_loss = running / config["val_every"]
            running = 0.0
            val_loss = evaluate_val(model, packed, val_ep, val_start, config["K"],
                                    config["H"], config["batch_size"], device, use_bf16)
            log.append({"step": step, "epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "elapsed_s": time.time() - t0})
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(os.path.join(out_dir, "best.pt"), model, config,
                                val_loss, step)
            print(f"step {step} ep {epoch} train {train_loss:.5f} val {val_loss:.5f} "
                  f"best {best_val:.5f} ({time.time() - t0:.0f}s)", flush=True)

    save_checkpoint(os.path.join(out_dir, "last.pt"), model, config,
                    float(log[-1]["val_loss"] if log else 0.0), step)
    dt = time.time() - t0
    result = {
        "name": config["name"],
        "n_params": n_params,
        "steps": step,
        "epochs": epoch,
        "best_val_loss": best_val,
        "final_train_loss": log[-1]["train_loss"] if log else None,
        "final_val_loss": log[-1]["val_loss"] if log else None,
        "seconds": dt,
        "steps_per_second": step / dt if dt > 0 else 0.0,
        "n_train_windows": len(train_ds),
    }
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1), flush=True)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the baseline next-frame model")
    ap.add_argument("--name", default=None)
    ap.add_argument("--packed", default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--H", type=int, default=None)
    ap.add_argument("--base", type=int, default=None)
    ap.add_argument("--conditioning", default=None)
    ap.add_argument("--batch", dest="batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--val-every", dest="val_every", type=int, default=None)
    ap.add_argument("--val-windows", dest="val_windows", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--train-fraction", dest="train_fraction", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config-json", default=None, help="optional JSON config file")
    args = ap.parse_args(argv)

    config = dict(DEFAULT_CONFIG)
    if args.config_json:
        with open(args.config_json) as f:
            config.update(json.load(f))
    # CLI overrides (only where the user actually passed a value)
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    config.update({k: v for k, v in overrides.items() if k != "config_json"})

    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
