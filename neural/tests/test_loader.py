"""Loader verification: alignment (item 2), rules (item 3), throughput + smoke (item 7)."""

import json
import os
import time

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from dataset import PackedDataset, WindowDataset


# --------------------------------------------------------------------------- #
# Item 2 — action alignment
# --------------------------------------------------------------------------- #

def test_alignment(packed96, src):
    rng = np.random.default_rng(0)
    eps = rng.choice(packed96.episodes, size=8, replace=False)
    K, H = 4, 1
    checked = 0
    for e in eps:
        m = json.load(open(src / e["source_path"] / "manifest.json"))
        frames = m["frames"]
        L = e["length"]
        starts = list(range(0, L - K - H + 1, max(1, L // 4)))
        for s in starts:
            w = packed96.window(int(e["episode_index"]), s, K, H)
            for j in range(K):
                g = s + j  # frame index within the episode
                fr = frames[g]
                next_fr = frames[g + 1]
                got_ax = w.context_action[j][0]
                got_ahc = w.context_action[j][1]

                exp_ax = fr["actionX"]
                if exp_ax is None:
                    assert np.isnan(got_ax)
                else:
                    # action paired with context frame j == manifest actionX[g]
                    assert np.isclose(got_ax, exp_ax, atol=1e-6)
                    # == manifest controlSignalX[g+1] (the schema's cross-check)
                    assert np.isclose(got_ax, next_fr["controlSignalX"], atol=1e-6)

                exp_ahc = fr["actionHasClicked"]
                if exp_ahc is None:
                    assert np.isnan(got_ahc)
                else:
                    assert got_ahc == (1.0 if exp_ahc else 0.0)
            checked += 1
    assert checked > 0


# --------------------------------------------------------------------------- #
# Item 3 — validity rules
# --------------------------------------------------------------------------- #

def _independent_valid_count(manifest, K, H, require_safe=True):
    """Count valid windows directly from a manifest (independent of the loader)."""
    frames = manifest["frames"]
    L = len(frames)
    if L < K + H:
        return 0
    unsafe = [bool(fr.get("viewUnsafe")) for fr in frames]
    W = K + H
    count = 0
    for s in range(L - W + 1):
        if require_safe and any(unsafe[s: s + W]):
            continue
        count += 1
    return count


def test_rules(packed96, src):
    K, H = 4, 1
    # no episode appears in two splits
    seen = {}
    for e in packed96.episodes:
        assert e["split"] in ("train", "val", "test")
        assert seen.get(e["episode_id"], e["split"]) == e["split"]
        seen[e["episode_id"]] = e["split"]

    for split in ("train", "val", "test"):
        ep, starts = packed96.enumerate_windows(split, K, H, require_safe=True)
        # every returned window is inside one episode and contains no unsafe frame
        for eid, s in zip(ep, starts):
            e = packed96.episodes[int(eid)]
            assert s + K + H <= e["length"]  # never crosses an episode boundary
            w = packed96.window(int(eid), int(s), K, H)
            assert not w.unsafe.any()  # no viewUnsafe frame

        # independent count from the manifests
        total = 0
        for e in packed96.episodes_by_split[split]:
            m = json.load(open(src / e["source_path"] / "manifest.json"))
            total += _independent_valid_count(m, K, H)
        assert total == len(ep), f"{split}: loader {len(ep)} vs manifests {total}"


# --------------------------------------------------------------------------- #
# Item 7 — throughput + smoke test
# --------------------------------------------------------------------------- #

def test_throughput(packed96):
    # The pilot's fast path keeps the (1.8 GB) split in RAM. A larger set (the
    # Stage 4B bulk at 96x72 is ~18 GB) does not fit, so an env knob lets the same
    # test measure the default memory-mapped mode; the default is unchanged.
    # Windows are tiny, so DataLoader worker IPC overhead dominates and
    # num_workers=0 is fastest here; measured and reported in the report.
    K = 4
    in_memory = os.environ.get("LNE_THROUGHPUT_IN_MEMORY", "1") == "1"
    mode = "in-memory" if in_memory else "memory-mapped"
    d = PackedDataset(packed96.path, in_memory=in_memory)
    ds = WindowDataset(d, "train", K=K, H=1, normalize=True)
    loader = DataLoader(ds, batch_size=64, num_workers=0, shuffle=False)
    t0 = time.time()
    count = 0
    for batch in loader:
        count += batch["context"].shape[0]
        if count >= 8192:
            break
    dt = time.time() - t0
    wps = count / dt
    print(f"\nthroughput: {wps:.0f} windows/s (K={K}, 96x72, {mode}, 1 process)")
    # In-memory keeps the pilot's floor; memory-mapped (an 18 GB set read from
    # disk in one process) is far slower, so it gets its own, honest floor.
    floor = float(os.environ.get("LNE_THROUGHPUT_MIN", "3000" if in_memory else "100"))
    assert wps > floor, f"loader too slow: {wps:.0f} windows/s (floor {floor:.0f})"


def test_smoke_train(packed96):
    """A few dozen steps of a trivial conv net fed by the loader, on the GPU."""
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    K, H = 4, 1
    ds = WindowDataset(packed96, "train", K=K, H=H, normalize=True)
    loader = DataLoader(ds, batch_size=32, num_workers=2, shuffle=True,
                        pin_memory=True)

    net = torch.nn.Sequential(
        torch.nn.Conv2d(3 * K, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 3 * H, 3, padding=1),
    ).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    losses = []
    it = iter(loader)
    for _ in range(40):
        batch = next(it)
        x = batch["context"].cuda(non_blocking=True)          # [B, K, 3, H, W]
        y = batch["target"].cuda(non_blocking=True)            # [B, H, 3, H, W]
        x = x.reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        y = y.reshape(y.shape[0], -1, y.shape[-2], y.shape[-1])
        opt.zero_grad()
        out = net(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    torch.cuda.synchronize()
    assert all(np.isfinite(l) for l in losses)
    # loss should trend down over a few dozen steps on a trivial net
    assert losses[-1] < losses[0]
    print(f"\nsmoke: loss {losses[0]:.4f} -> {losses[-1]:.4f} over 40 steps")
