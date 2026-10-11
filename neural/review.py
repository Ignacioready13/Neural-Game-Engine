"""Generate the one review image (windows.png) for the user's visual judgment.

Draws windows THROUGH the loader (never by reading PNGs directly): a first group
of training windows chosen where the action changes (just after a click), and a
second group of windows that were excluded because they contain a viewUnsafe
frame, with the unsafe frames marked.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import PackedDataset


def _find_action_change_windows(packed, K, H, n, max_per_episode=1):
    """Return (episode, target_start) windows where a click changes the action.

    A click accepted for step i changes actionX[i]; that action drives frame i ->
    i+1, so the turn begins at frame i+1. We make i+1 the *target* frame, i.e. the
    window's first context frame is `i+1-K` and its last context frame (i) carries
    the new actionX. Returns windows as (episode_index, context_start).
    """
    out = []
    per_episode = {}
    labels = packed.labels
    for e in packed.episodes:
        off = e["frame_offset"]
        L = e["length"]
        ax = labels["actionX"][off: off + L]
        unsafe = labels["viewUnsafe"][off: off + L]
        for i in range(1, L):
            # action change at step i (a click was accepted), not on final frame
            if np.isnan(ax[i]):
                continue
            if ax[i] == ax[i - 1]:
                continue
            target = i + 1
            start = target - K          # context = target-K .. target-1
            if start < 0 or target >= L:
                continue
            # window must be safe (else it would be excluded from training)
            if unsafe[start: start + K + H].any():
                continue
            if per_episode.get(e["episode_index"], 0) >= max_per_episode:
                continue
            per_episode[e["episode_index"]] = per_episode.get(e["episode_index"], 0) + 1
            out.append((e["episode_index"], int(start)))
            if len(out) >= n:
                return out
    return out


def _find_unsafe_windows(packed, K, H, n):
    """Return windows (which the default loader would drop) containing viewUnsafe,
    drawn from distinct episodes."""
    ep, start = packed.enumerate_windows("train", K, H, require_safe=False)
    out = []
    seen = set()
    for e, s in zip(ep, start):
        w = packed.window(int(e), int(s), K, H)
        if w.unsafe.any() and int(e) not in seen:
            out.append((int(e), int(s)))
            seen.add(int(e))
            if len(out) >= n:
                return out
    return out


def _draw_window_row(axs, w: "Window", title: str):
    """Draw one window (K context frames + H target frames) across `axs`."""
    n_frames = w.K + w.H
    for j in range(n_frames):
        ax = axs[j]
        ax.imshow(w.frames[j], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        is_target = j >= w.K
        is_unsafe = bool(w.unsafe[j])
        axx = w.actions[j][0]
        clicked = w.actions[j][1]
        axx_s = "null" if np.isnan(axx) else f"{axx:+.3f}"
        clicked_s = "null" if np.isnan(clicked) else ("T" if clicked >= 0.5 else "F")
        label = f"{'TGT ' if is_target else 'ctx '}{j}\naX={axx_s}\nclk={clicked_s}"
        ax.set_title(label, fontsize=7)
        if is_unsafe:
            for sp in ax.spines.values():
                sp.set_color("red")
                sp.set_linewidth(3)
        elif is_target:
            for sp in ax.spines.values():
                sp.set_color("green")
                sp.set_linewidth(2)


def make_review_image(packed_path: str, out_path: str, K: int = 4, H: int = 1):
    packed = PackedDataset(packed_path)
    good = _find_action_change_windows(packed, K, H, 6)
    bad = _find_unsafe_windows(packed, K, H, 2)

    rows = len(good) + len(bad)
    ncols = K + H
    fig, axs = plt.subplots(rows, ncols, figsize=(ncols * 1.35, rows * 1.6))
    if rows == 1:
        axs = axs[np.newaxis, :]

    r = 0
    for i, (e, s) in enumerate(good):
        w = packed.window(e, s, K, H)
        _draw_window_row(axs[r], w, f"window {i}")
        axs[r, 0].set_ylabel(f"OK {i+1}\nep{e} s{s}", fontsize=7)
        r += 1
    for i, (e, s) in enumerate(bad):
        w = packed.window(e, s, K, H)
        _draw_window_row(axs[r], w, f"excluded {i}")
        axs[r, 0].set_ylabel(f"UNSAFE {i+1}\nep{e} s{s}", fontsize=7, color="red")
        r += 1

    fig.suptitle(
        f"windows.png — {packed_path}\n"
        "Rows: K context frames (ctx) then target (TGT); aX=actionX, clk=actionHasClicked. "
        "Red border = viewUnsafe frame.",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    print(make_review_image(args.packed, args.out))
