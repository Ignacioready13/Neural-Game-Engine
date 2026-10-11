"""Measure model speed and resource usage (Stage 4 section 2.6).

Reports parameter count, forward FLOPs, precision, peak GPU memory for a
training step and for batch-1 inference, batch-1 inference latency/fps, and
autoregressive rollout-loop throughput. Compare the last number against the
15 fps interactive target from the brief.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import torch

from dataset import PackedDataset
from eval_harness import rollout
from model import build_model, count_parameters, load_checkpoint


def _flops(model, ctx, act):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fcm:
            model.predict(ctx, act)
        return int(fcm.get_total_flops())
    except Exception:
        return None


def _latency_ms(model, ctx, act, n=200, warmup=20):
    for _ in range(warmup):
        model.predict(ctx, act)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(ctx, act)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    packed = PackedDataset(args.packed, in_memory=True)
    model, payload = load_checkpoint(args.checkpoint, device)
    cfg = payload["model_config"]
    K = cfg["K"]
    res = packed.images.shape[1:3]  # (H, W)

    n_params = count_parameters(model)
    precision = payload.get("config", {}).get("mixed_precision", "fp32")

    # representative batch-1 context/action and a batch-`batch` for the train step
    ep, start = packed.enumerate_windows("val", K, 1, require_safe=True)
    w = packed.window(int(ep[0]), int(start[0]), K, 1)
    ctx1 = (torch.from_numpy(w.context.transpose(0, 3, 1, 2)[None]).float().to(device)
            / 127.5 - 1.0)
    act1 = torch.from_numpy(w.context_action[None]).float().to(device)
    ctx_b = ctx1.repeat(args.batch, 1, 1, 1, 1)
    act_b = act1.repeat(args.batch, 1, 1)

    model.eval()
    with torch.no_grad():
        flops = _flops(model, ctx1, act1)

    # peak memory: train-like (forward + backward) then reset, then inference
    torch.cuda.reset_peak_memory_stats()
    model.train()
    out = model.predict(ctx_b, act_b)
    out.sum().backward()
    torch.cuda.synchronize()
    peak_train_mb = torch.cuda.max_memory_allocated() / 1e6

    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model.predict(ctx1, act1)
    torch.cuda.synchronize()
    peak_infer_mb = torch.cuda.max_memory_allocated() / 1e6

    lat_ms = _latency_ms(model, ctx1, act1)

    # rollout-loop throughput
    t0 = time.perf_counter()
    with torch.no_grad():
        rollout(packed, model, int(ep[0]), int(start[0]), K, 40, device=device)
    torch.cuda.synchronize()
    roll_s = time.perf_counter() - t0
    roll_fps = 40 / roll_s

    result = {
        "resolution": list(res),
        "n_params": n_params,
        "flops_forward_batch1": flops,
        "precision": precision,
        "peak_train_mem_mb": round(peak_train_mb, 1),
        "peak_inference_mem_mb": round(peak_infer_mb, 1),
        "inference_batch1_latency_ms": round(lat_ms, 3),
        "inference_batch1_fps": round(1000 / lat_ms, 1),
        "rollout_loop_fps": round(roll_fps, 1),
        "fps_target": 15,
        "rollout_fps_vs_target": round(roll_fps / 15, 2),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
