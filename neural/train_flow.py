"""Stage 5 — train the stochastic generator (Milestone 2).

Trains ``flow.FlowSteerModel`` on top of a Milestone-1 steering checkpoint: the
readout + warp are frozen (the steering path), and the residual rectified-flow net
learns the newly revealed content. Windows come from bulk_v1 (all safe training
windows, one-step supervision), with turn-onset windows oversampled and mirror
augmentation; the conditioning context is noise-augmented so the generator stays
robust to its own (imperfect) frames at play time.

Run (from /neural):

    ./memcap.sh --mem 14G -- .venv/bin/python train_flow.py --name flow64 \
        --packed data/bulk_v1_64x48 --core runs/steer64/best.pt --max-steps 6000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dataset import PackedDataset
from flow import FlowSteerModel, flow_training_loss, save_flow_checkpoint
from steer import click_features
from train import _set_seed

EXTRA_HIST = 8
SELF_K = 6              # self-conditioning roll-in steps (train on own outputs)

DEFAULT_CONFIG = {
    "name": "flow64",
    "packed": "data/bulk_v1_64x48",
    "core": "runs/steer64/best.pt",
    "K": 4,
    "base": 32,
    "steps": 4,
    "cond_dim": 128,
    "core_mode": "warp_resid",
    "freeze_core": True,
    "batch_size": 64,
    "lr": 5e-4,
    "weight_decay": 0.0,
    "max_steps": 6000,
    "seed": 0,
    "mixed_precision": "bf16",
    "noise_aug": 0.05,
    "self_prob": 0.5,       # fraction of a batch trained on the model's own frames
    "noise_scale": 0.08,    # flow initial-noise scale (matched to the residual)
    "self_k": SELF_K,
    "attn": False,          # low-resolution spatial self-attention in the denoiser
    "net_base": None,       # denoiser width (defaults to `base`, which sizes the core)
    "edge_weight": 0.0,     # up-weight the revealed strip the warp had to invent
    "self_k_min": None,     # if set, roll in at least this many steps
    "mirror_prob": 0.5,
    "onset_weight": 3.0,
    "val_every": 500,
    "log_every": 50,        # progress line with steps/s (0 disables)
    "val_windows": 256,
    "val_seeds": 3,         # sampling seeds averaged for the checkpoint decision
    "num_workers": 3,
    "resume": None,         # checkpoint path to continue training from
    "out": "runs",
}


class SteerWindowDataset(Dataset):
    """Windows with the extended action history the click features need."""

    def __init__(self, packed: PackedDataset, split: str, K: int,
                 require_safe: bool = True, self_k: int = SELF_K):
        self.packed = packed
        self.K = K
        self.self_k = self_k
        self.ep, self.start = packed.enumerate_windows(split, K, 1, require_safe)
        self.ep = self.ep.astype(np.int64)
        self.start = self.start.astype(np.int64)

    def __len__(self):
        return int(len(self.ep))

    def __getitem__(self, i):
        e = int(self.ep[i]); s = int(self.start[i])
        rec = self.packed.episodes[e]
        off, L = rec["frame_offset"], rec["length"]
        K = self.K
        ctx = np.asarray(self.packed.images[off + s: off + s + K])       # [K,h,w,3]
        n_ex = self.self_k + 1
        hi = min(off + s + K + n_ex, off + L)
        nxt = np.asarray(self.packed.images[off + s + K: hi])            # [n_ex,h,w,3]
        if len(nxt) < n_ex:
            pad = n_ex - len(nxt)
            nxt = np.concatenate([nxt, np.repeat(nxt[-1:], pad, axis=0)], axis=0)
        lo = max(0, s - EXTRA_HIST)
        hist = self.packed.action_vectors(
            self.packed.labels[off + lo: off + s + K + self.self_k])
        hist = np.nan_to_num(hist, nan=0.0).astype(np.float32)
        want = EXTRA_HIST + K + self.self_k
        if len(hist) < want:
            pad = want - len(hist)
            hist = np.concatenate([np.repeat(hist[:1], pad, axis=0), hist], axis=0)
        return {
            "context": torch.from_numpy(np.ascontiguousarray(
                ctx.transpose(0, 3, 1, 2))).float() / 127.5 - 1.0,      # [K,3,h,w]
            "target": torch.from_numpy(np.ascontiguousarray(
                nxt[0].transpose(2, 0, 1))).float() / 127.5 - 1.0,      # [3,h,w]
            "target_k": torch.from_numpy(np.ascontiguousarray(
                nxt.transpose(0, 3, 1, 2))).float() / 127.5 - 1.0,      # [n_ex,3,h,w]
            "history": torch.from_numpy(np.ascontiguousarray(hist)),    # [EXTRA+K+k,2]
        }


def onset_flags(packed, ep, start, K):
    """True where a click change event lands within the last 3 context steps.

    Vectorized per episode (a per-window Python loop over ~700k windows is far too
    slow when it has to touch the memory-mapped labels)."""
    out = np.zeros(len(ep), dtype=bool)
    order = np.argsort(ep, kind="stable")
    ep_s, st_s = ep[order], start[order]
    bounds = np.searchsorted(ep_s, np.arange(len(packed.episodes)))
    for ei, rec in enumerate(packed.episodes):
        lo = bounds[ei]; hi = bounds[ei + 1] if ei + 1 < len(bounds) else len(ep_s)
        if lo >= hi:
            continue
        off, L = rec["frame_offset"], rec["length"]
        a = np.nan_to_num(packed.action_vectors(packed.labels[off: off + L]), nan=0.0)
        ch = np.zeros(L, dtype=np.int64)
        if L > 1:
            ch[1:] = (np.abs(np.diff(a[:, 0])) + np.abs(np.diff(a[:, 1])) > 1e-6)
        cch = np.concatenate([[0], np.cumsum(ch)])
        s = st_s[lo:hi]
        t = s + K - 1
        r = cch[np.minimum(t, L - 1)] - cch[np.clip(t - 3, 0, L - 1)]
        out[order[lo:hi]] = r > 0
    return out


def _prep(batch, cfg, device, rng):
    ctx = batch["context"].to(device)               # [B,K,3,h,w]
    tgt = batch["target"].to(device)                # [B,3,h,w]
    tgt_k = batch["target_k"].to(device)            # [B,n_ex,3,h,w]
    hist = batch["history"].numpy()
    if cfg["mirror_prob"] > 0:
        flip = rng.random(hist.shape[0]) < cfg["mirror_prob"]
        if flip.any():
            fi = np.where(flip)[0]
            ctx[fi] = torch.flip(ctx[fi], dims=[4])
            tgt_k[fi] = torch.flip(tgt_k[fi], dims=[4])
            tgt[fi] = torch.flip(tgt[fi], dims=[2])
            hist[flip, :, 0] *= -1.0
    hist_t = torch.from_numpy(np.ascontiguousarray(hist)).to(device)
    return ctx, tgt, tgt_k, hist_t


def feats_at(hist_t, K, k):
    """Click features for the K frames starting ``k`` steps into the history tail."""
    return click_features(hist_t)[:, EXTRA_HIST + k: EXTRA_HIST + k + K]


@torch.no_grad()
def evaluate(model, packed, ds, cfg, device, use_bf16, seeds=(0,)):
    """Validation flow loss and sampled L1, averaged over ``seeds`` sampling seeds.

    ``val_sample_l1`` is the sampled one-step error and it is *stochastic*: with one
    seed it moves by a factor of two between neighbouring checkpoints (flow96's log
    swings 0.068 → 0.286 → 0.097), so selecting ``best.pt`` on a single-seed value
    picks noise.  Averaging over seeds makes the checkpoint choice meaningful; the
    per-seed standard deviation is reported alongside it.
    """
    model.eval()
    rng = np.random.default_rng(0)
    n = min(cfg["val_windows"], len(ds))
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    tot_flow, m = 0.0, 0
    per_seed = {int(sd): 0.0 for sd in seeds}
    loader = DataLoader(torch.utils.data.Subset(ds, idx.tolist()),
                        batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    for batch in loader:
        ctx, tgt, tgt_k, hist_t = _prep(batch, {**cfg, "mirror_prob": 0.0}, device, rng)
        feats = feats_at(hist_t, cfg["K"], 0)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            loss, _ = flow_training_loss(model, ctx, feats, tgt, noise_aug=0.0)
            outs = {int(sd): model.predict(ctx, features=feats, seed=int(sd))
                    for sd in seeds}
        tot_flow += float(loss.item()) * ctx.shape[0]
        for sd, out in outs.items():
            per_seed[sd] += float(F.l1_loss(out.float(), tgt)) * ctx.shape[0]
        m += ctx.shape[0]
    model.train()
    vals = np.array([per_seed[int(sd)] / m for sd in seeds], dtype=np.float64)
    return {"val_flow": tot_flow / m, "val_sample_l1": float(vals.mean()),
            "val_sample_l1_sd": float(vals.std()), "val_sample_l1_seeds": len(seeds)}


def train(config: dict) -> dict:
    _set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = config["mixed_precision"] == "bf16" and device == "cuda"
    rng = np.random.default_rng(config["seed"])

    packed = PackedDataset(config["packed"], in_memory=False)
    train_ds = SteerWindowDataset(packed, "train", config["K"],
                                  self_k=int(config.get("self_k", SELF_K)))
    val_ds = SteerWindowDataset(packed, "val", config["K"],
                                self_k=int(config.get("self_k", SELF_K)))
    print(f"train windows {len(train_ds)}, val {len(val_ds)}", flush=True)

    model = FlowSteerModel(K=config["K"], H=10, base=config["base"],
                           steps=config["steps"], cond_dim=config["cond_dim"],
                           core_mode=config["core_mode"],
                           freeze_core=config["freeze_core"],
                           noise_scale=config.get("noise_scale", 0.08),
                           attn=bool(config.get("attn", False)),
                           net_base=config.get("net_base")).to(device)
    if config["core"]:
        core_payload = torch.load(config["core"], map_location=device, weights_only=False)
        model.core.load_state_dict(core_payload["state_dict"])
        print(f"loaded steering core from {config['core']} "
              f"(step {core_payload.get('step')}, val {core_payload.get('val_loss'):.4f})",
              flush=True)
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    onset = onset_flags(packed, train_ds.ep, train_ds.start, config["K"])
    w = np.ones(len(train_ds), np.float64)
    w[onset] = 1.0 + config["onset_weight"]
    sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=len(w), replacement=True)
    loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler,
                        num_workers=config["num_workers"], pin_memory=(device == "cuda"),
                        drop_last=True, persistent_workers=(config["num_workers"] > 0))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])
    start_step = 0
    if config.get("resume"):
        ck = torch.load(config["resume"], map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        start_step = int(ck.get("step", 0))
        print(f"resumed from {config['resume']} at step {start_step} "
              f"(model weights only; the optimizer state is not kept)", flush=True)
    out_dir = os.path.join(config["out"], config["name"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**config, "resumed_from": config.get("resume"),
                   "n_params_trainable": n_params,
                   "n_train_windows": len(train_ds), "n_onset": int(onset.sum()),
                   "device": device}, f, indent=1)

    log, best, step, t0 = [], float("inf"), int(start_step), time.time()
    running, seen = 0.0, 0
    it = iter(loader)
    while step < config["max_steps"]:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        ctx, tgt, tgt_k, hist_t = _prep(batch, config, device, rng)
        # self-conditioning: for part of the batch, first roll the model forward a
        # few steps on the *true* click history and train on those frames, so the
        # generator stays coherent on its own (imperfect) inputs. Without it the
        # closed loop drifts and the measured shift collapses.
        sk = int(config.get("self_k", SELF_K))
        self_rows = np.where(rng.random(ctx.shape[0]) < config["self_prob"])[0]
        if sk > 0 and len(self_rows):
            sel = torch.from_numpy(self_rows).to(device)
            k_lo = int(config.get("self_k_min") or 1)
            n_steps = int(rng.integers(min(k_lo, sk), sk + 1))
            with torch.no_grad():
                buf = ctx[sel].clone()
                for j in range(n_steps):
                    nxt = model.predict(
                        buf, features=feats_at(hist_t, config["K"], j)[sel],
                        seed=int(rng.integers(0, 10 ** 6)))
                    buf = torch.cat([buf[:, 1:], nxt[:, None]], dim=1)
            k_used = n_steps
            ctx = ctx.clone()
            ctx[sel] = buf.detach()
            tgt = tgt.clone()
            tgt[sel] = tgt_k[sel, k_used].detach()
            feats = feats_at(hist_t, config["K"], 0)
            feats = feats.clone()
            feats[sel] = feats_at(hist_t, config["K"], k_used)[sel]
        else:
            feats = feats_at(hist_t, config["K"], 0)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            loss, _ = flow_training_loss(model, ctx, feats, tgt,
                                         noise_aug=config["noise_aug"],
                                         edge_weight=float(config.get("edge_weight") or 0.0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        running += float(loss.item()); seen += 1; step += 1
        if config.get("log_every") and step % config["log_every"] == 0:
            el = time.time() - t0
            print(f"  step {step}/{config['max_steps']}  {step / max(1e-9, el):.2f} steps/s  "
                  f"elapsed {el:.0f}s  train_flow {running / max(1, seen):.4f}", flush=True)
        if step % config["val_every"] == 0 or step == config["max_steps"]:
            tr = running / seen; running, seen = 0.0, 0
            v = evaluate(model, packed, val_ds, config, device, use_bf16,
                         seeds=tuple(range(int(config.get("val_seeds", 1) or 1))))
            rec = {"step": step, "train_flow": tr, "elapsed_s": time.time() - t0, **v}
            log.append(rec)
            print(f"step {step} train {tr:.4f} val_flow {v['val_flow']:.4f} "
                  f"val_sample_l1 {v['val_sample_l1']:.4f}"
                  f"+-{v.get('val_sample_l1_sd', 0.0):.4f} ({time.time() - t0:.0f}s)",
                  flush=True)
            if v["val_sample_l1"] < best:
                best = v["val_sample_l1"]
                save_flow_checkpoint(os.path.join(out_dir, "best.pt"), model, config,
                                     best, step)
            save_flow_checkpoint(os.path.join(out_dir, "last.pt"), model, config,
                                 v["val_sample_l1"], step)
    dt = time.time() - t0
    result = {"name": config["name"], "n_params_trainable": n_params, "steps": step,
              "best_val_sample_l1": best, "seconds": dt, "steps_per_second": step / dt,
              "log": log[-1] if log else {}, "n_train_windows": len(train_ds)}
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1), flush=True)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the Stage 5 generator (M2)")
    for k, dv in DEFAULT_CONFIG.items():
        if k in ("out",):
            continue
        if isinstance(dv, bool):
            t = lambda s: s.lower() in ("1", "true", "yes")  # noqa: E731
        elif isinstance(dv, int):
            t = int
        elif isinstance(dv, float):
            t = float
        else:
            t = str
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, default=None, type=t)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config-json", default=None)
    args = ap.parse_args(argv)
    config = dict(DEFAULT_CONFIG)
    if args.config_json:
        with open(args.config_json) as f:
            config.update(json.load(f))
    config.update({k: v for k, v in vars(args).items()
                   if v is not None and k != "config_json"})
    # Guard against the easy mistake of passing the *training* budget to --steps,
    # which is the number of Euler steps in the sampler: 8000 of those means 8000
    # U-Net evaluations per generated frame and a run that never reaches step 1.
    if int(config["steps"]) > 64:
        raise SystemExit(
            f"--steps {config['steps']} is the ODE sampling-steps count, not the "
            f"training budget; use --max-steps {config['steps']} for training steps")
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
