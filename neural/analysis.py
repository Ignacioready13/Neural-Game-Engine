"""Read-only access to the packed analysis sidecar (`analysis.jsonl`).

The sidecar carries ground-truth quantities that are NOT model inputs but are
useful for evaluation: player pose/heading, angular velocity, the per-frame
doorway candidates (id, screenX, reachable), persona, and so on. It is keyed by
``(episode, frame)``; this module maps those keys onto the packed dataset's
*global* frame index (``episode.frame_offset + frame``) so the harness can slice
the sidecar exactly like it slices ``images.bin``.

This is additive: it does not touch the Stage 3 loader or packer, and it is
lazily loaded only by the evaluation harness.
"""

from __future__ import annotations

import json
import os

import numpy as np


class Analysis:
    """Vectorized + per-frame views of the analysis sidecar for one packed set."""

    def __init__(self, packed):
        self.packed = packed
        self.heading = np.full(packed.num_frames, np.nan, dtype=np.float32)
        self.omega = np.full(packed.num_frames, np.nan, dtype=np.float32)
        # per-frame doorway candidates (list of {doorwayId, screenX, reachable, screenDistance})
        self.candidates = [None] * packed.num_frames

        path = os.path.join(packed.path, "analysis.jsonl")
        offsets = {e["episode_index"]: e["frame_offset"] for e in packed.episodes}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                g = offsets[row["episode"]] + row["frame"]
                if "playerHeading" in row and row["playerHeading"] is not None:
                    self.heading[g] = float(row["playerHeading"])
                if "playerOmega" in row and row["playerOmega"] is not None:
                    self.omega[g] = float(row["playerOmega"])
                cand = row.get("candidates")
                if cand is not None:
                    self.candidates[g] = [
                        {"doorwayId": c.get("doorwayId"),
                         "screenX": float(c.get("screenX")),
                         "reachable": bool(c.get("reachable")),
                         "screenDistance": float(c.get("screenDistance"))}
                        for c in cand
                    ]

    def heading_at(self, episode: int, frame: int) -> float:
        """Player heading at a given (episode, frame), global-indexed."""
        e = self.packed.episodes[episode]
        return float(self.heading[e["frame_offset"] + frame])

    def heading_change(self, episode: int, frame: int) -> float:
        """Signed heading change over the step ``frame -> frame+1``, wrapped to ±pi."""
        a = self.heading_at(episode, frame)
        b = self.heading_at(episode, frame + 1)
        d = (b - a + np.pi) % (2 * np.pi) - np.pi
        return float(d)

    def candidates_at(self, episode: int, frame: int) -> list[dict]:
        e = self.packed.episodes[episode]
        cand = self.candidates[e["frame_offset"] + frame]
        return cand if cand is not None else []


def load_analysis(packed) -> Analysis:
    return Analysis(packed)
