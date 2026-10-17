"""Stage 5c — the coherence-first generator trainer (time-budgeted, EMA, self-checking).

This is `train_flow.py`'s loop with three Stage 5c additions, all of them driven by
what `STAGE5C-PROGRESS.md` §4.0a/§4.0b found:

* **GameNGen-style noise-level-conditioned frame corruption** (``--noise-level-max``,
  ``--level-buckets``, ``--ctx-dropout``) — see ``flow.flow_training_loss``.
* **An EMA of the weights** used for every evaluation, snapshot and health check.
* **A wall-clock budget** (``--budget-min``): the learning rate is constant for the
  first 70 % of the budget and cosine-decays to ~7 % by the end, and the run finishes
  on time whatever the throughput is. Inside the process it saves a fresh snapshot
  every ``--snapshot-min`` minutes (never overwriting one), runs a short live-path
  health rollout on the EMA weights, appends one line to ``runs/<name>/health.jsonl``
  and writes one small strip for the user to look at later. No heavier testing.

Launched detached and wrapped in ``memcap.sh`` + ``powerlog.py``:

    ./memcap.sh --mem 12G -- .venv/bin/python powerlog.py \
        --run-id flow96e --purpose train --parent-run flow96c -- \
        .venv/bin/python train_flow5c.py --name flow96e --budget-min 200 --detach-ready
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from flow import (FlowSteerModel, flow_training_loss, save_flow_checkpoint)
from train import _set_seed
from train_flow import SteerWindowDataset, _prep, evaluate, feats_at, onset_flags
from dataset import PackedDataset

import stage5c_rollout as R

DEFAULT_CONFIG = {
    "name": "flow96e",
    "packed": "data/bulk_v1_64x48",
    "core": "runs/steer96/best.pt",
    "K": 4,
    "base": 32,
    "net_base": 64,
    "attn": True,
    "steps": 4,
    "cond_dim": 128,
    "core_mode": "warp_resid",
    "freeze_core": True,
    "batch_size": 48,
    "lr": 4e-4,
    "weight_decay": 0.0,
    "max_steps": 200000,      # a ceiling; the budget is normally the wall clock
    "budget_min": 0.0,        # >0 -> time-budgeted run (the main run)
    "lr_flat_frac": 0.7,      # constant LR for this fraction of the budget
    "lr_min_frac": 0.07,      # final LR as a fraction of the peak
    "ema_decay": 0.999,
    "noise_aug": 0.0,         # Stage 5's (dead) fixed aug; the real knob is below
    "noise_level_max": 0.5,
    "level_buckets": 10,
    "level_clean_frac": 0.25,
    "ctx_dropout": 0.1,
    "corrupt_context": False,
    "self_prob": 0.25,
    "self_k": 4,
    "self_k_min": 2,
    "noise_scale": 0.08,
    "mirror_prob": 0.5,
    "onset_weight": 3.0,
    "seed": 0,
    "mixed_precision": "bf16",
    "grad_clip": 1.0,
    "val_every": 1000,
    "val_windows": 512,
    "val_seeds": 3,
    "log_every": 100,
    "num_workers": 3,
    "snapshot_min": 40.0,
    "health_steps": 300,
    "health_contexts": 3,
    "status_min": 5.0,
    "restart_frac": 0.5,
    "max_restarts": 2,
    "out": "runs",
    "register": True,
    "final_name": None,       # defaults to <name>_final
}


class EMA:
    """A moving average of the model's floating-point state (weights + buffers)."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))
        model.load_state_dict(sd)

    def state_dict(self):
        return self.shadow


def _build(config, device):
    model = FlowSteerModel(
        K=config["K"], H=10, base=config["base"], steps=config["steps"],
        cond_dim=config["cond_dim"], core_mode=config["core_mode"],
        freeze_core=config["freeze_core"], noise_scale=config["noise_scale"],
        attn=bool(config["attn"]), net_base=config["net_base"],
        noise_level_max=config["noise_level_max"],
        level_buckets=config["level_buckets"],
        ctx_dropout=config["ctx_dropout"]).to(device)
    return model


def _lr_at(frac: float, cfg, base_lr: float) -> float:
    if frac <= cfg["lr_flat_frac"]:
        return base_lr
    x = min(1.0, (frac - cfg["lr_flat_frac"]) / max(1e-9, 1.0 - cfg["lr_flat_frac"]))
    lo = cfg["lr_min_frac"]
    return base_lr * (lo + (1.0 - lo) * 0.5 * (1.0 + np.cos(np.pi * x)))


def health_check(model, seeds, cfg, real_hf, out_dir, tag):
    """Short live-path no-click health rollout on the given (EMA) weights."""
    rows = []
    sheets = []
    for sd in seeds:
        fr = R.rollout_model(model, sd, steps=int(cfg["health_steps"]),
                             device=next(model.parameters()).device, K=cfg["K"])
        h = R.health(fr, real_hf=real_hf)
        h["seed"] = sd.label
        h["in_band"] = R.in_band(h)["all"]
        rows.append(h)
        p = os.path.join(out_dir, f"stage5c_{tag}_{sd.label.replace(' ', '').replace('@', 'at')}.png")
        try:
            R.contact_sheet(fr, p, (1, 20, 60, 120, 200), scale=3)
            sheets.append(os.path.relpath(p))
        except Exception as e:                                # pragma: no cover
            print(f"[health] sheet failed: {e}", flush=True)
    keys = ("frame_change", "hf_ratio", "contrast", "luma_mean", "near_black")
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    mean["in_band"] = float(np.mean([r["in_band"] for r in rows]))
    return {"contexts": rows, "mean": mean, "sheets": sheets}


def write_status(path, **kw):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    kw["when"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(kw, fh)


def train(config: dict) -> dict:
    _set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = config["mixed_precision"] == "bf16" and device == "cuda"
    rng = np.random.default_rng(config["seed"])

    out_dir = os.path.join(config["out"], config["name"])
    snap_dir = os.path.join(out_dir, "snapshots")
    for d in (out_dir, snap_dir):
        os.makedirs(d, exist_ok=True)
    status_path = os.path.join(out_dir, "status.json")
    health_path = os.path.join(out_dir, "health.jsonl")
    review_dir = os.path.join("review", "stage5c")

    packed = PackedDataset(config["packed"], in_memory=False)
    train_ds = SteerWindowDataset(packed, "train", config["K"],
                                  self_k=int(config["self_k"]))
    val_ds = SteerWindowDataset(packed, "val", config["K"], self_k=int(config["self_k"]))
    print(f"[5c] train windows {len(train_ds)}, val {len(val_ds)}", flush=True)

    model = _build(config, device)
    if config["core"]:
        ck = torch.load(config["core"], map_location=device, weights_only=False)
        model.core.load_state_dict(ck["state_dict"])
        print(f"[5c] steering core from {config['core']} (step {ck.get('step')})",
              flush=True)
    ema_model = _build(config, device)
    ema_model.load_state_dict(model.state_dict())
    ema = EMA(model, config["ema_decay"])
    n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    onset = onset_flags(packed, train_ds.ep, train_ds.start, config["K"])
    w = np.ones(len(train_ds), np.float64)
    w[onset] = 1.0 + config["onset_weight"]
    sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=len(w),
                                    replacement=True)
    loader = DataLoader(train_ds, batch_size=config["batch_size"], sampler=sampler,
                        num_workers=config["num_workers"], pin_memory=(device == "cuda"),
                        drop_last=True, persistent_workers=(config["num_workers"] > 0))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])

    real_hf = R.real_reference("data/bulk_v1_96x72", n_clips=8, steps=300)["stats"]["hf_energy"]
    h_seeds = R.pick_seeds(int(config["health_contexts"]), "val", offset=0)


    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**config, "n_params_trainable": n_params,
                   "n_train_windows": len(train_ds), "n_onset": int(onset.sum()),
                   "device": device}, f, indent=1)

    budget_min = float(config["budget_min"] or 0.0)
    budget_s = budget_min * 60.0 if budget_min > 0 else None
    t0 = time.time()
    step = 0
    running, seen = 0.0, 0
    best = float("inf")
    restarts = 0
    last_snapshot_s = 0.0
    last_status_s = 0.0
    history = []
    it = iter(loader)
    lr = float(config["lr"])
    stopped_reason = "max_steps"

    while True:
        elapsed = time.time() - t0
        frac = (elapsed / budget_s) if budget_s else (step / config["max_steps"])
        if budget_s and elapsed >= budget_s:
            stopped_reason = "budget"
            break
        if not budget_s and step >= config["max_steps"]:
            break
        lr = _lr_at(frac, config, float(config["lr"]))
        for g in opt.param_groups:
            g["lr"] = lr
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        ctx, tgt, tgt_k, hist_t = _prep(batch, config, device, rng)
        sk = int(config["self_k"])
        self_rows = np.where(rng.random(ctx.shape[0]) < config["self_prob"])[0]
        if sk > 0 and len(self_rows):
            sel = torch.from_numpy(self_rows).to(device)
            k_lo = int(config.get("self_k_min") or 1)
            n_steps = int(rng.integers(min(k_lo, sk), sk + 1))
            with torch.no_grad():
                buf = ctx[sel].clone()
                for j in range(n_steps):
                    nxt = model.predict(buf, features=feats_at(hist_t, config["K"], j)[sel],
                                        seed=int(rng.integers(0, 10 ** 6)),
                                        level=0)
                    buf = torch.cat([buf[:, 1:], nxt[:, None]], dim=1)
            ctx = ctx.clone()
            ctx[sel] = buf.detach()
            tgt = tgt.clone()
            tgt[sel] = tgt_k[sel, n_steps].detach()
            feats = feats_at(hist_t, config["K"], 0)
            feats = feats.clone()
            feats[sel] = feats_at(hist_t, config["K"], n_steps)[sel]
        else:
            feats = feats_at(hist_t, config["K"], 0)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            loss, _ = flow_training_loss(
                model, ctx, feats, tgt, noise_aug=config["noise_aug"],
                noise_level_max=config["noise_level_max"],
                level_buckets=config["level_buckets"],
                level_clean_frac=config["level_clean_frac"],
                ctx_dropout=config["ctx_dropout"],
                corrupt_context=bool(config["corrupt_context"]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(config["grad_clip"]))
        opt.step()
        ema.update(model)
        lv = float(loss.item())
        running += lv
        seen += 1
        step += 1

        bad = not np.isfinite(lv)
        if bad and restarts < int(config["max_restarts"]):
            restarts += 1
            snap = sorted(s for s in os.listdir(snap_dir) if s.startswith("snap_"))
            base_lr = float(config["lr"]) * (float(config["restart_frac"]) ** restarts)
            print(f"[5c] non-finite loss at step {step}: restart #{restarts} "
                  f"(lr {base_lr:g}, from {snap[-1] if snap else 'scratch'})", flush=True)
            config["lr"] = base_lr
            if snap:
                payload = torch.load(os.path.join(snap_dir, snap[-1]), map_location=device,
                                     weights_only=False)
                model.load_state_dict(payload["state_dict"])
                ema_model.load_state_dict(payload["state_dict"])
                ema = EMA(model, config["ema_decay"])
            else:
                model = _build(config, device)
                ema = EMA(model, config["ema_decay"])
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                   lr=base_lr, weight_decay=config["weight_decay"])
            running, seen = 0.0, 0
            with open(health_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"event": "restart", "step": step,
                                     "when": time.strftime("%H:%M:%S"),
                                     "lr": base_lr}) + "\n")
            continue

        if config["log_every"] and step % int(config["log_every"]) == 0:
            print(f"[5c] step {step} frac {frac:.3f} lr {lr:.2e} "
                  f"{step / max(1e-9, elapsed):.2f} steps/s elapsed {elapsed:.0f}s "
                  f"train_flow {running / max(1, seen):.4f}", flush=True)

        now = time.time()
        snap_due = (now - t0 - last_snapshot_s) >= float(config["snapshot_min"]) * 60.0
        if snap_due:
            last_snapshot_s = now - t0
            rec = {"step": step, "elapsed_s": round(elapsed, 1), "frac": round(frac, 4),
                   "lr": lr, "train_flow": running / max(1, seen)}
            ema.copy_to(ema_model)
            v = evaluate(ema_model, packed, val_ds, config, device, use_bf16,
                         seeds=tuple(range(int(config["val_seeds"]))))
            rec.update(v)
            h = health_check(ema_model, h_seeds, config, real_hf, review_dir,
                             f"{config['name']}_snap{step:07d}")
            rec["health"] = h["mean"]
            rec["health_sheets"] = h["sheets"]
            rec["health_ok"] = bool(h["mean"].get("in_band", 0) >= 2 / 3)
            snap_path = os.path.join(snap_dir, f"snap_{step:07d}.pt")
            if os.path.exists(snap_path):
                raise SystemExit(f"refusing to overwrite {snap_path}")
            save_flow_checkpoint(snap_path, ema_model, config, v["val_sample_l1"], step,
                                 extra={"ema": True, "health": h["mean"]})
            save_flow_checkpoint(os.path.join(out_dir, "last.pt"), ema_model, config,
                                 v["val_sample_l1"], step, extra={"ema": True})
            if v["val_sample_l1"] < best:
                best = v["val_sample_l1"]
                save_flow_checkpoint(os.path.join(out_dir, "best.pt"), ema_model, config,
                                     best, step, extra={"ema": True, "health": h["mean"]})
            history.append(rec)
            with open(health_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            with open(os.path.join(out_dir, "train_log.json"), "w") as fh:
                json.dump(history, fh, indent=1)
            print(f"[5c] SNAPSHOT step {step} val_flow {v['val_flow']:.4f} "
                  f"val_l1 {v['val_sample_l1']:.4f} health {json.dumps({k: round(x, 3) for k, x in h['mean'].items()})}",
                  flush=True)

        if (time.time() - t0 - last_status_s) >= float(config["status_min"]) * 60.0:
            last_status_s = time.time() - t0
            write_status(status_path, state="training", step=step, frac=round(frac, 4),
                         elapsed_s=round(elapsed, 1), steps_per_s=round(step / max(1e-9, elapsed), 3),
                         lr=lr, train_flow=round(running / max(1, seen), 4), restarts=restarts)

    # ---- finish: final EMA checkpoint + health, then register ------------------ #
    print(f"[5c] stopping ({stopped_reason}) at step {step}", flush=True)
    ema.copy_to(ema_model)
    final_path = os.path.join(out_dir, "final.pt")
    v = evaluate(ema_model, packed, val_ds, config, device, use_bf16,
                 seeds=tuple(range(int(config["val_seeds"]))))
    h = health_check(ema_model, h_seeds, config, real_hf, review_dir,
                     f"{config['name']}_final")
    save_flow_checkpoint(final_path, ema_model, config, v["val_sample_l1"], step,
                         extra={"ema": True, "final": True, "health": h["mean"]})
    rec = {"step": step, "elapsed_s": round(time.time() - t0, 1), "final": True,
           "lr": lr, "stopped_reason": stopped_reason, "restarts": restarts, **v,
           "health": h["mean"], "health_sheets": h["sheets"],
           "health_ok": bool(h["mean"].get("in_band", 0) >= 2 / 3)}
    with open(health_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    result = {"name": config["name"], "steps": step, "seconds": time.time() - t0,
              "steps_per_second": step / max(1e-9, time.time() - t0),
              "stopped_reason": stopped_reason, "restarts": restarts,
              "best_val_sample_l1": best, "val_flow": v["val_flow"],
              "val_sample_l1": v["val_sample_l1"], "health": h["mean"],
              "health_ok": rec["health_ok"], "n_params_trainable": n_params,
              "snapshots": len(history), "n_train_windows": len(train_ds)}
    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    write_status(status_path, state="finished", step=step,
                 stopped_reason=result["stopped_reason"], restarts=restarts,
                 health_ok=result["health_ok"],
                 val_sample_l1=result["val_sample_l1"])

    if config.get("register", True):
        fname = config.get("final_name") or f"{config['name']}_final"
        cmd = [sys.executable, "register_model.py", "--name", fname,
               "--generator", os.path.relpath(final_path),
               "--core", config["core"] or "runs/steer96/best.pt",
               "--parent", "flow96c", "--not-selected",
               "--description", f"{fname}: Stage 5c coherence run "
                                f"({config['name']}, {step} steps, GameNGen-style "
                                f"noise-conditioned context, EMA, budget={budget_min:g} min)",
               "--notes", json.dumps({"health": h["mean"], "val_sample_l1": v["val_sample_l1"]}),
               "--packed", "data/bulk_v1_96x72"]
        print("[5c] " + " ".join(cmd), flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        print(p.stdout.strip() or p.stderr.strip(), flush=True)
        result["register_rc"] = p.returncode
        result["register_out"] = (p.stdout or p.stderr).strip()
    print(json.dumps(result, indent=1), flush=True)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 5c coherence trainer")
    for k, dv in DEFAULT_CONFIG.items():
        if k in ("out",):
            continue
        if isinstance(dv, bool):
            t = lambda s: s.lower() in ("1", "true", "yes")  # noqa: E731
        elif k in ("net_base", "final_name", "restart_frac"):
            t = str
        elif isinstance(dv, int):
            t = int
        elif isinstance(dv, float):
            t = float
        else:
            t = str
        ap.add_argument(f"--{k.replace('_', '-')}", dest=k, default=None, type=t)
    ap.add_argument("--config-json", default=None)
    args = ap.parse_args(argv)
    config = dict(DEFAULT_CONFIG)
    if args.config_json:
        with open(args.config_json) as f:
            config.update(json.load(f))
    config.update({k: v for k, v in vars(args).items()
                   if v is not None and k != "config_json"})
    if int(config["steps"]) > 64:
        raise SystemExit("--steps is the sampler's ODE-step count, not a training budget")
    for k in ("net_base", "restart_frac"):
        config[k] = float(config[k]) if k == "restart_frac" else int(config[k])
    train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
