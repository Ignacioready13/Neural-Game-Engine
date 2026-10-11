"""P6a: oracle-intent input on the hold-last horizon model.

P5 showed the turn direction lives in the bottleneck (auxiliary head 0.89) but the
pixel-L1 decoder never maps it into the frames (counterfactual flat). P6 asks
*which* failure that is: getting direction INTO the generator, or the generator's
inability to express a small global view shift under a pixel loss?

P6a answers the second half directly: give the hold-last frame model the
**ground-truth intent** from the sidecar as an explicit input, and ask whether the
frames then turn. The intent is the door the pilot is steering toward after the
click, encoded as its screen X:

  scalar       — screenX of the reachable door nearest to the click's actionX
                 (the same quantity P4's heading head is trained to predict),
                 appended to the FiLM conditioning MLP; + = door on the right;
  column map   — a Gaussian bump at that screenX column, an extra image channel
                 concatenated to every context frame (an image-aligned cue).

Train with the true values (hold-last, mirror). Evaluate (i) with true values and
(ii) counterfactually: intent = left door / right door / none. If the frames
follow the intent, the generator *can* express a directional shift given an
explicit signal — and the deployment path is the P4 readout feeding this input.

Run: .venv/bin/python train_intent.py --name intent10 --packed data/pilot_v1_64x48 \
       --horizon 10 --mirror --max-steps 6000
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
from metrics import estimate_shift_x
from model import ConvBlock, count_parameters
from train import _set_seed
from train_hold import _onset, _sample_batch


def _oracle_intent(sidecar, e, g, clickx):
    """screenX of the reachable door nearest to clickx (the P4 oracle); NaN if none."""
    cands = [c for c in sidecar.candidates_at(int(e), int(g)) if c["reachable"]]
    if not cands or abs(clickx) <= 1e-3:
        return float("nan")
    d = min(cands, key=lambda c: abs(c["screenX"] - clickx))
    return float(d["screenX"])


class HoldIntentModel(nn.Module):
    """Hold-last FiLM U-Net conditioned on (action, intent scalar, intent column map)."""

    def __init__(self, K: int, H: int, base: int = 32, cond_dim: int = 64):
        super().__init__()
        n_act = K + H
        self.K = K
        self.enc1 = ConvBlock(K * 4, base)                       # RGB + 1 column-map ch
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
            nn.Linear(n_act * 2 + 1, cond_dim), nn.ReLU(),   # +1 intent scalar
            nn.Linear(cond_dim, 2 * sum(self._film_chs)),
        )

    def _film(self, x, g, b):
        return x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]

    def _column_map(self, sx, H, W):
        xs = torch.linspace(-1.0, 1.0, W, device=sx.device)
        col = (sx + 1.0) / 2.0 * (W - 1)                       # [B]
        d = xs[None, :] - col[:, None]                          # [B, W]
        sigma = W / 12.0
        g = torch.exp(-(d * d) / (2.0 * sigma * sigma))         # [B, W]
        g = torch.where(torch.isnan(col[:, None]), torch.zeros_like(g), g)
        return g[:, None, None, :].expand(-1, 1, H, -1)         # [B, 1, H, W]

    def forward(self, context, action, intent):
        B = context.shape[0]
        Hh, W = context.shape[3], context.shape[4]
        sx = torch.nan_to_num(intent, nan=0.0)
        cmap = self._column_map(intent, Hh, W)                  # [B,1,H,W] (NaN -> flat)
        xin = torch.cat([context, cmap.unsqueeze(1).expand(-1, self.K, -1, -1, -1)], dim=2)
        x = xin.reshape(B, self.K * 4, Hh, W)
        cond = self.film_mlp(torch.cat([action.reshape(B, -1), sx[:, None]], dim=1))
        gamma, beta = torch.chunk(cond, 2, dim=1)
        c3, c2, c1 = self._film_chs
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e3f = self._film(e3, gamma[:, :c3], beta[:, :c3])
        d2 = self.dec2(torch.cat([self.up2(e3f), e2], dim=1))
        d2 = self._film(d2, gamma[:, c3:c3 + c2], beta[:, c3:c3 + c2])
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d1 = self._film(d1, gamma[:, c3 + c2:], beta[:, c3 + c2:])
        resid = self.out(d1)
        frame = (context[:, -1].float() + resid.float()).clamp(-1, 1)
        return frame


class HoldWarpModel(HoldIntentModel):
    """P6b: structured output = differentiable horizontal warp of the last frame
    (a predicted scalar shift) + a residual for newly revealed / local content."""

    def __init__(self, K: int, H: int, base: int = 32, cond_dim: int = 64,
                 max_shift: float = 16.0, warp_only: bool = False):
        super().__init__(K, H, base, cond_dim)
        self.max_shift = max_shift
        self.warp_only = warp_only
        self.shift_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(base * 4, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def _warp(self, img, d_px):
        B, C, Hh, W = img.shape
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, Hh, device=img.device),
            torch.linspace(-1, 1, W, device=img.device), indexing="ij")
        xg = xs - (2.0 * d_px[:, None, None] / W)      # + d_px = content right
        grid = torch.stack([xg, ys[None].expand(B, -1, -1)], dim=-1)
        return F.grid_sample(img, grid, mode="bilinear",
                             padding_mode="border", align_corners=True)

    def forward(self, context, action, intent):
        B = context.shape[0]
        Hh, W = context.shape[3], context.shape[4]
        sx = torch.nan_to_num(intent, nan=0.0)
        cmap = self._column_map(intent, Hh, W)
        xin = torch.cat([context, cmap.unsqueeze(1).expand(-1, self.K, -1, -1, -1)], dim=2)
        x = xin.reshape(B, self.K * 4, Hh, W)
        cond = self.film_mlp(torch.cat([action.reshape(B, -1), sx[:, None]], dim=1))
        gamma, beta = torch.chunk(cond, 2, dim=1)
        c3, c2, c1 = self._film_chs
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e3f = self._film(e3, gamma[:, :c3], beta[:, :c3])
        d = torch.tanh(self.shift_head(e3f)).squeeze(1) * self.max_shift
        d2 = self.dec2(torch.cat([self.up2(e3f), e2], dim=1))
        d2 = self._film(d2, gamma[:, c3:c3 + c2], beta[:, c3:c3 + c2])
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d1 = self._film(d1, gamma[:, c3 + c2:], beta[:, c3 + c2:])
        resid = self.out(d1)
        warped = self._warp(context[:, -1], d).float()
        frame = warped if self.warp_only else (warped + resid.float()).clamp(-1, 1)
        return frame


def _make_batch_intent(packed, sidecar, ep, start, K, H, device, no_intent=False):
    ctx, act, tgt, itn = [], [], [], []
    for e, s in zip(ep.tolist(), start.tolist()):
        e2 = packed.episodes[int(e)]
        off = e2["frame_offset"]
        fr = packed.images[off + s: off + s + K + H]
        ctx.append(fr[:K].transpose(0, 3, 1, 2))
        tgt.append(fr[K + H - 1].transpose(2, 0, 1))
        a = packed.action_vectors(packed.labels[off + s: off + s + K + H])
        a = np.nan_to_num(a, nan=0.0).astype(np.float32)
        clickx = float(a[K - 1, 0])
        a[K:] = a[K - 1]                                   # hold-last
        act.append(a)
        g = s + K - 1
        itn.append(float("nan") if no_intent else _oracle_intent(sidecar, int(e), int(g), clickx))
    ctx = torch.from_numpy(np.ascontiguousarray(np.stack(ctx))).float().to(device) / 127.5 - 1.0
    tgt = torch.from_numpy(np.ascontiguousarray(np.stack(tgt))).float().to(device) / 127.5 - 1.0
    act = torch.from_numpy(np.stack(act)).float().to(device)
    itn = torch.from_numpy(np.array(itn, dtype=np.float32)).to(device)
    return ctx, act, tgt, itn


def train(config):
    _set_seed(config["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K, H = config["K"], config["horizon"]

    packed = PackedDataset(config["packed"], in_memory=True)
    sidecar = load_analysis(packed)
    ep, start = packed.enumerate_windows("train", K, H, require_safe=True)
    ep = ep.astype(np.int64); start = start.astype(np.int64)
    onset = _onset(packed, sidecar, ep, start, K, H, config.get("onset", "click"))
    print(f"train windows {len(ep)}, onset {int(onset.sum())} "
          f"({onset.sum() / len(ep) * 100:.1f}%)", flush=True)

    model = (HoldWarpModel(K, H, config.get("base", 32),
                           warp_only=config.get("warp_only", False))
             if config.get("warp") else HoldIntentModel(K, H, config.get("base", 32))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"],
                           weight_decay=config["weight_decay"])
    out_dir = os.path.join(config["out"], config["name"])
    os.makedirs(out_dir, exist_ok=True)
    full_cfg = dict(config)
    json.dump({**full_cfg, "n_train_windows": len(ep), "n_onset": int(onset.sum()),
               "n_params": count_parameters(model)},
              open(os.path.join(out_dir, "config.json"), "w"), indent=1)

    rng = np.random.default_rng(config["seed"] + 99)
    B = config["batch_size"]
    t0 = time.time()
    for step in range(1, config["max_steps"] + 1):
        eb, sb = _sample_batch(ep, start, onset, B, rng)
        ctx, act, tgt, itn = _make_batch_intent(packed, sidecar, eb, sb, K, H, device,
                                                config.get("no_intent", False))
        if config.get("mirror"):
            m = torch.from_numpy(rng.random(B) < 0.5).to(device)
            idx = m.nonzero(as_tuple=False).squeeze(1)
            if len(idx):
                ctx[idx] = torch.flip(ctx[idx], dims=[-1])
                tgt[idx] = torch.flip(tgt[idx], dims=[-1])
                act[idx, :, 0] = -act[idx, :, 0]
                itn[idx] = -itn[idx]
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            frame = model(ctx, act, itn)
        loss = F.l1_loss(frame.float(), tgt)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == config["max_steps"]:
            print(f"step {step} loss {float(loss.item()):.5f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    torch.save({"state_dict": model.state_dict(), "config": full_cfg},
               os.path.join(out_dir, "best.pt"))
    return out_dir


def _predict_frame(packed, model, e, s, K, H, act, itn, device):
    off = packed.episodes[int(e)]["frame_offset"]
    fr = packed.images[off + s: off + s + K]
    ctx = torch.from_numpy(np.ascontiguousarray(fr.transpose(0, 3, 1, 2)[None])) \
        .float().to(device) / 127.5 - 1.0
    a = torch.from_numpy(act[None]).float().to(device)
    it_t = torch.tensor([itn], dtype=torch.float32, device=device)
    pred = model(ctx, a, it_t)[0]
    pu = pred.permute(1, 2, 0).clamp(-1, 1) * 0.5 + 0.5
    return (pu * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()


def _intent_ablation(packed, model, sidecar, K, H, max_windows=300):
    """Val click-onset windows: true vs zeroed vs shuffled intent (held click true)."""
    device = next(model.parameters()).device
    ep, start = packed.enumerate_windows("val", K, H, require_safe=True)
    ep = ep.astype(np.int64); start = start.astype(np.int64)
    onset = _onset(packed, sidecar, ep, start, K, H, "click")
    ep, start = ep[onset], start[onset]
    rng = np.random.default_rng(0)
    if len(ep) > max_windows:
        pick = rng.choice(len(ep), max_windows, replace=False)
        ep, start = ep[pick], start[pick]

    def score(itn_a):
        tot = 0.0
        with torch.no_grad():
            for e, s, iv in zip(ep.tolist(), start.tolist(), itn_a.tolist()):
                off = packed.episodes[int(e)]["frame_offset"]
                a = packed.action_vectors(packed.labels[off + s: off + s + K + H])
                a = np.nan_to_num(a, nan=0.0).astype(np.float32)
                a[K:] = a[K - 1]
                pu = _predict_frame(packed, model, e, s, K, H, a, float(iv), device)
                gt = packed.images[off + s + K + H - 1].astype(np.float32)
                tot += float(np.abs(pu.astype(np.float32) - gt).mean())
        return tot / len(ep)

    itn_true = []
    for e, s in zip(ep.tolist(), start.tolist()):
        off = packed.episodes[int(e)]["frame_offset"]
        clickx = float(packed.action_vectors(packed.labels[off + s + K - 1: off + s + K])[0, 0])
        itn_true.append(_oracle_intent(sidecar, int(e), int(s) + K - 1, clickx))
    itn_true = np.nan_to_num(np.array(itn_true, dtype=np.float32), nan=0.0)
    l_true = score(itn_true)
    l_zero = score(np.zeros_like(itn_true))
    perm = rng.permutation(len(itn_true))
    l_shuf = score(itn_true[perm])
    print(f"\n=== P6a intent ablation (val click-onset, n={len(ep)}) ===")
    print(f"H-step L1  true {l_true:.5f}  zeroed {l_zero:.5f}  shuffled {l_shuf:.5f}")
    return {"l1_true": l_true, "l1_zeroed": l_zero, "l1_shuffled": l_shuf}


def _counterfactual(packed, model, sidecar, K, H, contexts, no_intent=False):
    device = next(model.parameters()).device
    Lc, Rc, Nc, Li, Ri, Ni = [], [], [], [], [], []
    meta = []
    model.eval()
    with torch.no_grad():
        for e, s, xl, xr, dl, dr in contexts:
            off = packed.episodes[int(e)]["frame_offset"]
            fr = packed.images[off + s: off + s + K]
            last_u8 = fr[K - 1]
            lab = packed.labels[off + s: off + s + K]
            true_ctx = np.nan_to_num(packed.action_vectors(lab), nan=0.0).astype(np.float32)
            held = true_ctx[K - 1].copy()
            nv = float("nan")

            # combined (deployment-realistic "last" variant): click + matching intent
            for name, x, itv in (("left", xl, nv if no_intent else xl),
                                 ("right", xr, nv if no_intent else xr),
                                 ("none", float(held[0]), nv)):
                a = true_ctx.copy()
                if name != "none":
                    a[K - 1] = [x, 1.0]
                fut = np.full((H, 2), [a[K - 1, 0], a[K - 1, 1]], np.float32)
                full = np.concatenate([a[None], fut[None]], axis=1)
                pu = _predict_frame(packed, model, e, s, K, H, full, itv, device)
                ({"left": Lc, "right": Rc, "none": Nc})[name].append(
                    estimate_shift_x(last_u8, pu))

            # intent-only (action held identically, only intent varies)
            a = true_ctx.copy()
            fut = np.full((H, 2), [a[K - 1, 0], a[K - 1, 1]], np.float32)
            full = np.concatenate([a[None], fut[None]], axis=1)
            for name, itv in (("left", nv if no_intent else xl),
                              ("right", nv if no_intent else xr),
                              ("none", nv)):
                pu = _predict_frame(packed, model, e, s, K, H, full, itv, device)
                ({"left": Li, "right": Ri, "none": Ni})[name].append(
                    estimate_shift_x(last_u8, pu))

            gt_shift = estimate_shift_x(last_u8, packed.images[off + s + K + H - 1])
            meta.append((e, s, min(dl, dr), gt_shift))

    def report(tag, L, R, N):
        L, R, N = np.array(L), np.array(R), np.array(N)
        print(f"\n--- P6a counterfactual {tag} (n={len(L)}) ---")
        print(f"mean shift  left {L.mean():+.3f}  right {R.mean():+.3f}  none {N.mean():+.3f}")
        both = np.mean((L > N) & (N > R))
        print(f"left-correct {np.mean(L > N):.3f}  right-correct {np.mean(R < N):.3f}  "
              f"both {both:.3f}")
        gt = np.array([m[3] for m in meta])
        click = (np.abs(L - N).mean() + np.abs(R - N).mean()) / 2
        ratio = click / max(1e-9, np.abs(gt).mean())
        print(f"GT |shift| {np.abs(gt).mean():.3f}  predicted |shift| {click:.3f}  "
              f"ratio {ratio:.3f}")
        return {"both_correct": float(both), "ratio": float(ratio),
                "mean_L": float(L.mean()), "mean_R": float(R.mean()),
                "mean_N": float(N.mean())}

    r_combined = report("combined click+intent", Lc, Rc, Nc)
    r_intent = report("intent-only (fixed action)", Li, Ri, Ni)
    return {"combined": r_combined, "intent_only": r_intent}


def evaluate(packed, model, sidecar, K, H, no_intent=False):
    from probe_click import _select_contexts
    abl = _intent_ablation(packed, model, sidecar, K, H)
    contexts = _select_contexts(packed, sidecar, K, H, max_per_ep=2)
    print(f"\nqualifying counterfactual contexts: {len(contexts)}")
    contexts = contexts[:200]
    cf = _counterfactual(packed, model, sidecar, K, H, contexts, no_intent=no_intent)
    return {**abl, **cf}


def main(argv=None):
    from train import DEFAULT_CONFIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="intent10")
    ap.add_argument("--packed", default="data/pilot_v1_64x48")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--onset", choices=("click", "quiet"), default="click")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--warp", action="store_true", help="P6b: structured warp+residual output")
    ap.add_argument("--warp-only", action="store_true", help="warp with NO residual (shift is the only knob)")
    ap.add_argument("--no-intent", dest="no_intent", action="store_true",
                    help="drop the oracle-intent input (actionX conditioning only)")
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
    model = (HoldWarpModel(K, H, args.base, warp_only=args.warp_only)
             if args.warp else HoldIntentModel(K, H, args.base))
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"),
                                     map_location="cuda", weights_only=False)["state_dict"])
    model.to("cuda").eval()
    res = evaluate(packed, model, sidecar, K, H, no_intent=args.no_intent)
    json.dump({**config, **res}, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    print("\nresult:", json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
