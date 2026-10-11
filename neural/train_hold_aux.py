"""P5: auxiliary heading head on the hold-last horizon model.

P4 showed turn direction is extractable from pixels + actions with a *direct*
heading readout. P5 asks whether the same signal, added as a **training-only**
auxiliary head on the hold-last frame model, pushes direction into the *frame*
output (which is all the harness measures at play time).

The model is the hold-last FiLM U-Net of ``train_hold.py`` plus a small heading
head on the film-conditioned bottleneck. Combined loss = H-step frame L1 +
``aux_weight`` * heading MSE (net heading change over H steps, from the sidecar).
The heading head is dropped at inference; only its gradient to the shared encoder
is used. After training we run the hold-last onset ablation and the
``probe_click`` counterfactual to see how much of the effect reaches the frames.

Run: .venv/bin/python train_hold_aux.py --name aux10 --packed data/pilot_v1_64x48 \
       --horizon 10 --mirror --max-steps 6000 --aux-weight 10
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis import load_analysis
from dataset import PackedDataset
from model import ConvBlock, count_parameters
from train import _set_seed
from train_hold import _click_onset, _onset, _sample_batch, ablate_hold

_ONSET_AHEAD = np.deg2rad(0.5)


class HoldAuxModel(nn.Module):
    """FiLM U-Net (hold-last) + a training-only heading head on the bottleneck."""

    def __init__(self, K: int, H: int, base: int = 32, cond_dim: int = 64):
        super().__init__()
        n_act = K + H
        self.K = K
        self.enc1 = ConvBlock(K * 3, base)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1, bias=False)
        self.enc2 = ConvBlock(base * 2, base * 2)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1, bias=False)
        self.enc3 = ConvBlock(base * 4, base * 4)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(base * 4, base * 2, 3, padding=1, bias=False))
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                 nn.Conv2d(base * 2, base, 3, padding=1, bias=False))
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 3, 3, padding=1)

        self._film_chs = [base * 4, base * 2, base]
        self.film_mlp = nn.Sequential(
            nn.Linear(n_act * 2, cond_dim), nn.ReLU(),
            nn.Linear(cond_dim, 2 * sum(self._film_chs)),
        )
        # auxiliary heading head on the film-conditioned bottleneck
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(base * 4, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def _film(self, x, g, b):
        return x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]

    def forward(self, context, action):
        B = context.shape[0]
        x = context.reshape(B, self.K * 3, *context.shape[3:])
        cond = self.film_mlp(action.reshape(B, -1))
        gamma, beta = torch.chunk(cond, 2, dim=1)
        c3, c2, c1 = self._film_chs
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e3f = self._film(e3, gamma[:, :c3], beta[:, :c3])
        heading = self.head(e3f).squeeze(1)
        d2 = self.dec2(torch.cat([self.up2(e3f), e2], dim=1))
        d2 = self._film(d2, gamma[:, c3:c3 + c2], beta[:, c3:c3 + c2])
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d1 = self._film(d1, gamma[:, c3 + c2:], beta[:, c3 + c2:])
        resid = self.out(d1)
        frame = (context[:, -1].float() + resid.float()).clamp(-1, 1)
        return frame, heading

    def predict(self, context, action):
        return self.forward(context, action)[0]


def _make_batch_aux(packed, sidecar, ep, start, K, H, device):
    """Hold-last batch + net heading-change targets."""
    ctx, act, tgt, y = [], [], [], []
    for e, s in zip(ep.tolist(), start.tolist()):
        e2 = packed.episodes[int(e)]
        off = e2["frame_offset"]
        fr = packed.images[off + s: off + s + K + H]
        ctx.append(fr[:K].transpose(0, 3, 1, 2))
        tgt.append(fr[K + H - 1].transpose(2, 0, 1))
        a = packed.action_vectors(packed.labels[off + s: off + s + K + H])
        a = np.nan_to_num(a, nan=0.0).astype(np.float32)
        a[K:] = a[K - 1]                 # hold-last
        act.append(a)
        g = s + K - 1
        h0 = sidecar.heading_at(int(e), int(g))
        h1 = sidecar.heading_at(int(e), int(g) + H)
        dh = (h1 - h0 + np.pi) % (2 * np.pi) - np.pi if not (np.isnan(h0) or np.isnan(h1)) else 0.0
        y.append(float(dh))
    ctx = torch.from_numpy(np.ascontiguousarray(np.stack(ctx))).float().to(device) / 127.5 - 1.0
    tgt = torch.from_numpy(np.ascontiguousarray(np.stack(tgt))).float().to(device) / 127.5 - 1.0
    act = torch.from_numpy(np.stack(act)).float().to(device)
    y = torch.from_numpy(np.array(y, dtype=np.float32)).float().to(device)
    return ctx, act, tgt, y


def train(config):
    _set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K, H = config["K"], config["horizon"]
    aux_w = config.get("aux_weight", 10.0)

    packed = PackedDataset(config["packed"], in_memory=True)
    sidecar = load_analysis(packed)
    ep, start = packed.enumerate_windows("train", K, H, require_safe=True)
    ep = ep.astype(np.int64); start = start.astype(np.int64)
    onset = _onset(packed, sidecar, ep, start, K, H, config.get("onset", "click"))
    print(f"train windows {len(ep)}, onset {int(onset.sum())} "
          f"({onset.sum() / len(ep) * 100:.1f}%), aux_weight={aux_w}", flush=True)

    model = HoldAuxModel(K, H, config.get("base", 32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config["weight_decay"])
    out_dir = os.path.join(config["out"], config["name"])
    os.makedirs(out_dir, exist_ok=True)
    full_cfg = dict(config)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**full_cfg, "n_train_windows": len(ep), "n_onset": int(onset.sum()),
                   "n_params": count_parameters(model)}, f, indent=1)

    rng = np.random.default_rng(config["seed"] + 99)
    B = config["batch_size"]
    t0 = time.time()
    for step in range(1, config["max_steps"] + 1):
        eb, sb = _sample_batch(ep, start, onset, B, rng)
        ctx, act, tgt, y = _make_batch_aux(packed, sidecar, eb, sb, K, H, device)
        if config.get("mirror"):
            m = torch.from_numpy(rng.random(B) < 0.5).to(device)
            idx = m.nonzero(as_tuple=False).squeeze(1)
            if len(idx):
                ctx[idx] = torch.flip(ctx[idx], dims=[-1])
                tgt[idx] = torch.flip(tgt[idx], dims=[-1])
                act[idx, :, 0] = -act[idx, :, 0]
                y[idx] = -y[idx]
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            frame, heading = model(ctx, act)
        lf = F.l1_loss(frame.float(), tgt)
        lh = F.mse_loss(heading.float(), y)
        loss = lf + aux_w * lh
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == config["max_steps"]:
            print(f"step {step} frame {float(lf.item()):.5f} head {float(lh.item()):.6f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    torch.save({"state_dict": model.state_dict(), "config": full_cfg},
               os.path.join(out_dir, "best.pt"))
    return out_dir


def evaluate_aux(packed, model, sidecar, K, H, aux_w):
    device = next(model.parameters()).device
    from probe_click import _select_contexts, _run
    from metrics import estimate_shift_x

    # 1) onset ablation (held-click true/zeroed/shuffled) on H-step frame L1
    print(f"\n=== P5 onset ablation (hold-last, aux_weight={aux_w}) ===")
    ablate_hold(packed, model, sidecar, K, H, "val", 300, onset_kind="click")

    # 2) click counterfactual via probe_click (the "last" variant = deployment-realistic)
    contexts = _select_contexts(packed, sidecar, K, H, max_per_ep=2)
    print(f"\nqualifying counterfactual contexts: {len(contexts)}")
    contexts = contexts[:200]
    res, meta = _run(packed, model, sidecar, K, H, contexts)
    L = np.array(res["last"]["left"])
    R = np.array(res["last"]["right"])
    Hd = np.array(res["last"]["hold"])
    gt = np.array([m[3] for m in meta])
    print(f"\n--- P5 counterfactual ('last' variant, n={len(L)}) ---")
    print(f"mean shift  left {L.mean():+.3f}  right {R.mean():+.3f}  hold {Hd.mean():+.3f}")
    print(f"left-correct {np.mean(L > Hd):.3f}  right-correct {np.mean(R < Hd):.3f}  "
          f"both {np.mean((L > Hd) & (Hd > R)):.3f}")
    print(f"GT |shift| {np.abs(gt).mean():.3f}")
    click = (np.abs(L - Hd).mean() + np.abs(R - Hd).mean()) / 2
    ratio = click / max(1e-9, np.abs(gt).mean())
    print(f"predicted click effect |shift| {click:.3f}  predicted-vs-real ratio {ratio:.3f}")
    return {"both_correct": float(np.mean((L > Hd) & (Hd > R))),
            "left_correct": float(np.mean(L > Hd)), "right_correct": float(np.mean(R < Hd)),
            "mean_shift_L": float(L.mean()), "mean_shift_R": float(R.mean()),
            "ratio": float(ratio)}


def main(argv=None):
    from train import DEFAULT_CONFIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="aux10")
    ap.add_argument("--packed", default="data/pilot_v1_64x48")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--onset", choices=("click", "quiet"), default="click")
    ap.add_argument("--aux-weight", dest="aux_weight", type=float, default=10.0)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--batch", dest="batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args(argv)

    config = dict(DEFAULT_CONFIG)
    config.update({k: v for k, v in vars(args).items() if v is not None})
    config["weight_decay"] = 0.0
    config["conditioning"] = "film"
    config["n_action_frames"] = args.K + args.horizon
    config["hold_last"] = True

    packed = PackedDataset(args.packed, in_memory=True)
    sidecar = load_analysis(packed)
    K, H = args.K, args.horizon

    out_dir = train(config)
    model = HoldAuxModel(K, H, args.base)
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"),
                                     map_location="cuda", weights_only=False)["state_dict"])
    model.to("cuda").eval()
    res = evaluate_aux(packed, model, sidecar, K, H, args.aux_weight)
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump({**config, **res}, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
