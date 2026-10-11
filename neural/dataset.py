"""
Loader over a packed dataset (see packer.py) that serves training windows and
longer clips correctly and fast.

# Action alignment (read this first)

`actionX[i]` is the action that drives the transition from frame `i` to frame
`i+1`. The schema's action-alignment rule is therefore:

    the action paired with frame `i` is `actionX[i]` (= `controlSignalX[i+1]`).

For a training window with K context frames `t-K .. t-1` and H target frames
`t .. t+H-1`, the action paired with context frame `j` is `actionX[j]`, and the
action that drives into target frame `t+m` is `actionX[t+m-1]` (the previous
frame's action). This module returns each action as a float32 vector
`[actionX, actionHasClicked]`:

  - `actionX` in [-1, 1]; `NaN` only on an episode's final frame (which has no
    "next" transition and is never a context frame).
  - `actionHasClicked` in {0.0, 1.0}; `NaN` only on an episode's final frame.
  - Before the first accepted click the engine is in its "never clicked" default
    mode: `actionHasClicked == 0` and `actionX == 0`.

# Window validity

A window never crosses an episode boundary or a split. By default a window is
dropped if ANY frame it covers (context or target) has `viewUnsafe`; pass
`require_safe=False` to keep such windows (analysis only). By default a window
needs a full K frames of context (no padding); `pad_first=True` pads the start
of an episode by repeating frame 0. The episode's final frame (actionX null) can
only be a target: the length constraint already forbids it from being a context
frame.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

SPLITS = ("train", "val", "test")


@dataclass
class Window:
    """A contiguous K+H frame window from one episode."""
    episode: int            # global episode index
    start: int              # index of the first context frame within the episode
    frames: np.ndarray      # [K+H, H, W, 3] uint8
    actions: np.ndarray     # [K+H, 2] float32 [actionX, actionHasClicked], NaN = null
    unsafe: np.ndarray      # [K+H] bool
    K: int
    H: int

    @property
    def context(self) -> np.ndarray:
        return self.frames[: self.K]

    @property
    def target(self) -> np.ndarray:
        return self.frames[self.K:]

    @property
    def context_action(self) -> np.ndarray:
        return self.actions[: self.K]

    @property
    def target_action(self) -> np.ndarray:
        # action driving into target m is actionX at the frame before it
        return self.actions[self.K - 1: self.K - 1 + self.H]


def frames_to_tensor(frames: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """uint8 [N, H, W, 3] -> float32 [N, 3, H, W], in [-1, 1] when normalize else [0, 255]."""
    t = torch.from_numpy(np.ascontiguousarray(frames).astype(np.float32))
    t = t.permute(0, 3, 1, 2)
    if normalize:
        t = t / 127.5 - 1.0
    return t


class PackedDataset:
    """Memory-mapped view over a packed set with window/clip access."""

    def __init__(self, path: str, in_memory: bool = False):
        self.path = os.path.abspath(path)
        with open(os.path.join(self.path, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(os.path.join(self.path, "index.json"), "r", encoding="utf-8") as f:
            self.episodes = json.load(f)["episodes"]

        self.width, self.height = self.meta["resolution"]
        n, h, w, c = self.meta["image_shape"]
        self.num_frames = n
        self.label_dtype = np.dtype([tuple(f) for f in self.meta["label_fields"]])
        self.images = np.memmap(
            os.path.join(self.path, "images.bin"), mode="r", dtype=np.uint8,
            shape=(n, h, w, c),
        )
        self.labels = np.memmap(
            os.path.join(self.path, "labels.bin"), mode="r", dtype=self.label_dtype,
            shape=(n,),
        )
        if in_memory:
            # Load the packed arrays into RAM (fits for the pilot; ~1.8 GB at 96x72).
            self.images = np.asarray(self.images)
            self.labels = np.asarray(self.labels)
        self.episode_by_id = {e["episode_id"]: e for e in self.episodes}
        self.episodes_by_split = {s: [e for e in self.episodes if e["split"] == s]
                                  for s in SPLITS}

    # -- low-level access --------------------------------------------------- #

    def _episode_index(self, episode) -> int:
        if isinstance(episode, str):
            return self.episode_by_id[episode]["episode_index"]
        return int(episode)

    def _episode(self, episode) -> dict:
        return self.episodes[self._episode_index(episode)]

    def get_clip(self, episode, start_frame: int, length: int):
        """Contiguous L frames + labels from one episode (never crosses a boundary).

        Returns (frames [L, H, W, 3] uint8, labels structured [L]).
        """
        e = self._episode(episode)
        if start_frame < 0 or start_frame + length > e["length"]:
            raise ValueError(
                f"clip [{start_frame}, {start_frame + length}) exceeds episode "
                f"{e['episode_id']} of length {e['length']}"
            )
        off = e["frame_offset"]
        s = off + start_frame
        return self.images[s: s + length], self.labels[s: s + length]

    def action_vectors(self, labels: np.ndarray) -> np.ndarray:
        """Structured label rows -> [N, 2] float32 [actionX, actionHasClicked], NaN = null."""
        ax = labels["actionX"].astype(np.float32)
        ahc = labels["actionHasClicked"].astype(np.float32)
        ahc = np.where(ahc >= 0, ahc, np.float32(np.nan))
        return np.stack([ax, ahc], axis=-1)

    # -- window primitive --------------------------------------------------- #

    def window(self, episode, start: int, K: int, H: int, pad_first: bool = False) -> Window:
        """Build a K+H frame window. `start` is the first context frame's index
        within the episode (may be negative when pad_first=True).

        A window is contiguous frames, so the common (non-padded) path is a plain
        slice — no per-window fancy indexing."""
        e = self._episode(episode)
        off = e["frame_offset"]
        L = e["length"]
        if not pad_first:
            if start < 0 or start + K + H > L:
                raise ValueError(
                    f"window start {start} (K={K}, H={H}) out of range for episode "
                    f"{e['episode_id']} of length {L}"
                )
            frames = self.images[off + start: off + start + K + H]
            labels = self.labels[off + start: off + start + K + H]
        else:
            idx = np.clip(np.arange(start, start + K + H, dtype=np.int64), 0, L - 1)
            frames = self.images[off + idx]
            labels = self.labels[off + idx]
        actions = self.action_vectors(labels)
        unsafe = labels["viewUnsafe"].astype(bool)
        return Window(episode=self._episode_index(episode), start=int(start),
                      frames=np.asarray(frames), actions=actions, unsafe=unsafe, K=K, H=H)

    # -- window enumeration ------------------------------------------------- #

    def enumerate_windows(self, split: str, K: int, H: int, require_safe: bool = True,
                          pad_first: bool = False):
        """Return (win_episode int64[N], win_start int64[N]) for all valid windows."""
        win_episodes = []
        win_starts = []
        W = K + H
        for e in self.episodes_by_split[split]:
            L = e["length"]
            if L < W:
                continue
            max_start = L - W
            starts = np.arange(max_start + 1, dtype=np.int64)  # 0 .. max_start (full context)
            if pad_first:
                starts = np.arange(-(K - 1), max_start + 1, dtype=np.int64)
            if require_safe:
                off = e["frame_offset"]
                unsafe = self.labels["viewUnsafe"][off: off + L].astype(np.int64)
                # sliding count of unsafe frames in each length-W span
                cum = np.concatenate([[0], np.cumsum(unsafe)])
                if pad_first:
                    # padded windows read from frame 0 onward; a window at start s<0
                    # covers frames max(0,s) .. s+W-1 (with the left edge repeated),
                    # but its effective span length is W regardless.
                    counts = np.array([
                        _unsafe_span(unsafe, s, W) for s in starts
                    ])
                else:
                    counts = cum[W:] - cum[:-W]
                valid = counts == 0
                starts = starts[valid]
            win_episodes.append(np.full(len(starts), e["episode_index"], dtype=np.int64))
            win_starts.append(starts)
        if win_episodes:
            return (np.concatenate(win_episodes), np.concatenate(win_starts))
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64))

    def count_valid_windows(self, split: str, K: int, H: int, require_safe: bool = True,
                            pad_first: bool = False) -> int:
        ep, _ = self.enumerate_windows(split, K, H, require_safe, pad_first)
        return int(len(ep))


def _unsafe_span(unsafe: np.ndarray, start: int, W: int) -> int:
    """Number of unsafe frames a (possibly padded) window at `start` actually covers."""
    lo = max(start, 0)
    hi = min(start + W, len(unsafe))
    return int(unsafe[lo:hi].sum())


# --------------------------------------------------------------------------- #
# PyTorch Dataset / iteration
# --------------------------------------------------------------------------- #

class WindowDataset(torch.utils.data.Dataset):
    """Map-style dataset of training windows over one split.

    Yields dicts with `context` [K,3,H,W] float, `context_action` [K,2],
    `target` [H,3,H,W], `target_action` [H,2], and `episode`/`start` indices.
    `normalize=True` returns images in [-1, 1] (diffusion convention), else [0,255].
    """

    def __init__(self, packed: PackedDataset, split: str, K: int, H: int = 1,
                 require_safe: bool = True, pad_first: bool = False,
                 normalize: bool = True):
        self.packed = packed
        self.split = split
        self.K = K
        self.H = H
        self.normalize = normalize
        self.require_safe = require_safe
        self.pad_first = pad_first
        self.win_episode, self.win_start = packed.enumerate_windows(
            split, K, H, require_safe, pad_first
        )

    def __len__(self) -> int:
        return int(len(self.win_episode))

    def __getitem__(self, idx: int) -> dict:
        ep = int(self.win_episode[idx])
        start = int(self.win_start[idx])
        w = self.packed.window(ep, start, self.K, self.H, self.pad_first)
        return {
            "context": frames_to_tensor(w.context, self.normalize),
            "context_action": torch.from_numpy(np.ascontiguousarray(w.context_action)),
            "target": frames_to_tensor(w.target, self.normalize),
            "target_action": torch.from_numpy(np.ascontiguousarray(w.target_action)),
            "episode": ep,
            "start": start,
        }


def iter_windows(packed: PackedDataset, split: str, K: int, H: int = 1,
                 require_safe: bool = True, pad_first: bool = False):
    """Deterministic, ordered generator over valid windows (for evaluation)."""
    ep, start = packed.enumerate_windows(split, K, H, require_safe, pad_first)
    for e, s in zip(ep, start):
        yield packed.window(int(e), int(s), K, H, pad_first)
