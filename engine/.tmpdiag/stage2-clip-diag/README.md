# Stage 2 clipping diagnosis (archived scratch)

Read-only diagnosis of the "camera see-through" frame the user flagged in
`_review/transitions.png` (`h2->h4 @ seed 0 f133`). No engine behaviour was
changed. Run from `engine/`:

```
node node_modules/tsx/dist/cli.mjs .tmpdiag/stage2-clip-diag/diag-clip.ts 150 0 -1
node node_modules/tsx/dist/cli.mjs .tmpdiag/stage2-clip-diag/probe-pulse.ts
```

## `diag-clip.ts`

Replays a pilot episode renderer-free (World + Steering + navigation +
`resolveWallCollision`, mirroring `Engine.step`), validates the replay pose
against the manifest per frame, and measures, per frame, the point-to-triangle
distance from the camera to every triangle of every live entity. Also reports a
screen-space estimate of the area lost to the near plane (Sutherland-Hodgman
clip against a virtual 1 mm near minus the real near).

Replay fidelity: **max pose error 1.4e-14** over 148 of 149 pilot episodes.
One episode (`seed 78`) diverges after f91 (a debounce-boundary flip) and is
excluded from the statistics.

## `probe-pulse.ts`

Paired pixel test that the click pulse is **not** in captured frames: for 400
click events it compares frame *i+1* to frame *i* at the pulse disc vs a control
band. Result: disc-minus-control Δlum = **−0.43** → no pulse.

## Findings (full pilot, 149 episodes / 88,047 frames)

- Near plane is **0.02** (`src/render/scene.ts`); min camera-to-geometry distance
  ever observed is **0.0014**.
- **39 frames (0.044%)** come within the near plane, across **20 / 149 episodes
  (13.4%)**.
- 15 of those are within 5 mm; 4 within 2 mm.
- Clusters at doorway approaches: 21 of 39 sit ~1.4–1.8 u from a doorway centre;
  the rest are two multi-frame wall-slide runs in `seed 55`.
- Two mechanisms: (i) **near-plane clip of a front-facing wall** (most frames,
  e.g. `seed 0 f133`, min distance 0.0196); (ii) **back-face see-through** —
  the camera ends up behind a *neighbouring* entity's wall plane, which is
  front-side-only and so culled (~11 frames, e.g. `seed 111 f175`).
- Why the failsafe missed it: L1 `wallOverlap` only fires when the player leaves
  the floor. Here the player stays on the floor (just very close to a wall), so
  no flag is raised.

## Outcome (2026-10-11)

The fix landed as a **label, not a change**: a per-frame `viewUnsafe` boolean
(`src/sim/viewSafety.ts`, wired into `Engine.getState()`), threshold = camera
near (0.02) + `VIEW_UNSAFE_MARGIN` (0.10). No simulation or rendering change; the
near plane stays 0.02. Margin 0.10 was chosen so `seed 0`'s whole graze flags
(f129–f135, spanning f132/f133/f134; f128/f136 clear).

The regenerated pilot measures **357 frames (0.41%) over 90 / 150 episodes** —
the diag's independent geometric count over the 149 non-diverged episodes was
**333** at the same threshold, and `+ seed 78`'s 24 (excluded here for the f91
divergence) makes **357**. The two measurements agree exactly. See
`docs/STAGE2-REPORT.md` §4b.
