"""Stage 5 — the Milestone 4 measurements for the neuromorphic / hardware write-up.

Measures, on the final 96×72 model, everything the brief asks for (§3 Milestone 4):
parameter count, multiply-adds per generated frame with each component's share,
numeric precision, latency breakdown per component, activation sparsity after the
nonlinearities, and the frame-to-frame sparsity of real and generated frames.

    ./memcap.sh --mem 12G -- .venv/bin/python hw_analysis.py \
        --packed data/bulk_v1_96x72 --checkpoint runs/flow96/best.pt \
        --out eval_out/hw_flow96.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from dataset import PackedDataset, frames_to_tensor
from eval_harness import features_from_history, model_predict
from gates import _u8, load_model, onset_windows
from metrics import estimate_shift_x
from shift_table import load_target_table
from steer import click_features


def count_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def flops_of(fn, *args, **kw):
    """Multiply-adds (FLOPs / 2) of one call, via the torch flop counter."""
    from torch.utils.flop_counter import FlopCounterMode
    fc = FlopCounterMode(display=False)
    with fc:
        fn(*args, **kw)
    return int(fc.get_total_flops() / 2)


class ReluSparsity:
    """Fraction of exact zeros and near-zeros in the outputs of every ReLU."""

    def __init__(self, model):
        self.total = 0
        self.zeros = 0
        self.near = 0
        self.handles = [
            m.register_forward_hook(self._hook)
            for m in model.modules() if isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU))
        ]

    def _hook(self, mod, inp, out):
        o = out.detach()
        self.total += o.numel()
        self.zeros += int((o == 0).sum().item())
        self.near += int((o.abs() < 1e-3).sum().item())

    def close(self):
        for h in self.handles:
            h.remove()

    def result(self):
        if not self.total:
            return {"exact_zero": None, "near_zero": None, "n": 0}
        return {"exact_zero": self.zeros / self.total,
                "near_zero": self.near / self.total, "n": self.total}


def frame_change_sparsity(a_u8, b_u8, thresholds=(2, 10, 30)):
    """Share of pixels changing by more than each threshold (0-255 units)."""
    d = np.abs(a_u8.astype(np.int16) - b_u8.astype(np.int16)).max(axis=-1)
    return {str(t): float((d > t).mean()) for t in thresholds}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 5 hardware / sparsity measurements")
    ap.add_argument("--packed", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    packed = PackedDataset(args.packed)
    model, payload = load_model(args.checkpoint, device)
    K = int(payload["model_config"]["K"])
    table = load_target_table(packed, args.packed, "heading")
    wins = onset_windows(packed, table, K)[:16]
    e, s = wins[0]
    rec = packed.episodes[e]
    off = rec["frame_offset"]
    ctx = frames_to_tensor(np.asarray(packed.images[off + s: off + s + K]))[None].to(device)
    h = np.nan_to_num(packed.action_vectors(
        packed.labels[off + max(0, s - 8): off + s + K]), nan=0.0).astype(np.float32)
    hist = np.concatenate([np.repeat(h[:1], 8, axis=0), h], axis=0)[-(8 + K):]
    feats = click_features(torch.from_numpy(hist[None]).float().to(device))[:, -K:]

    core = model.core
    den = model.net
    out = {
        "resolution": [packed.width, packed.height],
        "checkpoint": args.checkpoint,
        "n_params": {"readout+core": count_params(core), "denoiser": count_params(den),
                     "total": count_params(model), "trainable": int(sum(
                         p.numel() for p in model.parameters() if p.requires_grad))},
        "precision": "float32 (training uses bf16 autocast)",
        "sampling_steps": model.steps,
    }

    # ---- multiply-adds per generated frame -------------------------------- #
    with torch.no_grad():
        readout_macs = flops_of(core.readout, ctx, feats)
        cond_dim = feats.shape[1] * feats.shape[2] + model.H + 1
        den_macs = flops_of(den, torch.randn(1, 8, packed.height, packed.width,
                                            device=device),
                            torch.randn(1, cond_dim, device=device))
        # the warp is a grid_sample: 4 gathers + blends per output pixel
        warp_macs = int(3 * packed.width * packed.height * 4)
        total_macs = readout_macs + model.steps * den_macs + warp_macs
        out["macs_per_frame"] = {
            "readout": readout_macs, "denoiser_per_step": den_macs,
            "denoiser_total": model.steps * den_macs, "warp": warp_macs,
            "total": total_macs,
            "share_readout": readout_macs / total_macs,
            "share_denoiser": model.steps * den_macs / total_macs,
            "share_warp": warp_macs / total_macs,
            "dense_fraction": 1.0,   # every op is a dense conv/matmul or a gather
        }

    # ---- latency breakdown ------------------------------------------------- #
    def timeit(fn, n=50):
        with torch.no_grad():
            for _ in range(5):
                fn()
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            if device == "cuda":
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n * 1000.0

    with torch.no_grad():
        t_readout = timeit(lambda: core.readout(ctx, feats))
        dx = core.readout(ctx, feats)[0][:, 0]
        from steer import warp_translate
        t_warp = timeit(lambda: warp_translate(ctx[:, -1], dx))
        warped = warp_translate(ctx[:, -1], dx)
        cond = torch.randn(1, cond_dim, device=device)
        x = torch.randn(1, 3, packed.height, packed.width, device=device)
        t_den = timeit(lambda: den(torch.randn(1, 8, packed.height, packed.width,
                                               device=device), cond))
        t_decode = timeit(lambda: _u8(warped[0]))
        frames = ctx.clone()
    out["latency_ms"] = {
        "readout": t_readout, "warp": t_warp, "denoiser_per_step": t_den,
        "denoiser_total": model.steps * t_den, "decode": t_decode,
        "sum_of_parts": t_readout + t_warp + model.steps * t_den + t_decode,
        "note": ("compare with the measured end-to-end loop in the G4 gate "
                 "(14.7 ms mean per frame incl. click features and the rollout bookkeeping)"),
    }

    # ---- activation sparsity ---------------------------------------------- #
    def sparsity_of(module, fn):
        rs = ReluSparsity(module)
        with torch.no_grad():
            fn()
        rs.close()
        return rs.result()

    out["activation_sparsity"] = {
        "readout": sparsity_of(core.readout, lambda: core.readout(ctx, feats)),
        "denoiser": sparsity_of(den, lambda: den(
            torch.randn(1, 8, packed.height, packed.width, device=device), cond)),
    }

    # ---- frame-to-frame sparsity of real and generated frames -------------- #
    real_sp, gen_sp = [], []
    with torch.no_grad():
        for e_i, s_i in wins[:8]:
            r = packed.episodes[e_i]
            o = r["frame_offset"]
            real = np.asarray(packed.images[o + s_i + K - 1: o + s_i + K - 1 + 3])
            real_sp.append({k: v for k, v in frame_change_sparsity(
                real[0], real[1]).items()})
            frames = frames_to_tensor(np.asarray(
                packed.images[o + s_i: o + s_i + K])).to(device)
            hh = np.nan_to_num(packed.action_vectors(
                packed.labels[o + max(0, s_i - 8): o + s_i + K]), nan=0.0).astype(np.float32)
            hh = np.concatenate([np.repeat(hh[:1], 8, axis=0), hh], axis=0)[-(8 + K):]
            acts = np.concatenate([np.repeat(hh[-1:], 2, axis=0)], axis=0)
            hist = [row for row in hh]
            prev_u8 = None
            for t in range(2):
                hist.append(acts[t])
                hist = hist[-(8 + K + 2):]
                fts = click_features(torch.from_numpy(
                    np.stack(hist[-8 - K:]))[None].float().to(device))[:, -K:]
                nxt = model_predict(model, frames[-K:].unsqueeze(0), None, fts, t)
                nxt = nxt[0] if isinstance(nxt, tuple) else nxt
                u = _u8(nxt[0])
                if prev_u8 is not None:
                    gen_sp.append(frame_change_sparsity(prev_u8, u))
                prev_u8 = u
                frames = torch.cat([frames, nxt], dim=0)
    def mean_sp(list_of):
        if not list_of:
            return {}
        return {k: float(np.mean([d[k] for d in list_of])) for k in list_of[0]}
    out["frame_to_frame_sparsity"] = {
        "real": mean_sp(real_sp), "generated": mean_sp(gen_sp),
        "definition": "share of pixels whose max channel change exceeds the threshold (0-255)",
        "real_mean_abs_change": float(np.mean([abs(
            np.asarray(packed.images[packed.episodes[w0]["frame_offset"] + w1 + K - 1],
                       dtype=np.float32).mean() -
            np.asarray(packed.images[packed.episodes[w0]["frame_offset"] + w1 + K],
                       dtype=np.float32).mean()) / 255.0 for w0, w1 in wins[:8]])),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
