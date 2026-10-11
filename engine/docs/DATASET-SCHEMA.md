# Dataset Schema (schema v2)

This is the **definitive data contract** for the headless dataset. It is written
for the future Python/PyTorch author who consumes these files and has never seen
this repository. The playtest recorder (`npm run record`) is debug tooling and
is **outside** this contract — it may write extra fields and is not guaranteed
to stay compatible.

The reader should not need to read any TypeScript to consume a dataset. Where
this file and the code disagree, treat the code as truth and file a correction.

---

## 1. Directory layout

A **run plan** (a JSON file, see §8) names a dataset. Generated episodes land in:

```
<outputRoot>/<planName>/
  <planName>.plan.json        # the plan that produced this dataset (copy kept alongside)
  aborted.jsonl               # one JSON line per aborted episode (see §7)
  train/seed_<seed>/          # episodes, grouped by split
    frames/000000.png         # lossless PNG, OUTPUT_WIDTH x OUTPUT_HEIGHT
    manifest.json             # per-episode metadata + per-frame records
  val/seed_<seed>/...
  test/seed_<seed>/...
  report.json                 # diagnostics report (see engine/src/headless/report.ts)
  report.md
```

- Splits are `train`, `val`, `test`. Each episode's seed is drawn from a **disjoint
  range** per split (see §8), so no seed appears in two splits.
- Episode directories are **deterministic** (derived from split + seed), never
  random UUIDs, so a batch is resumable and findable.
- Every frame is a PNG (lossless), never JPEG.

## 2. Timing and the fixed timestep

- The simulation advances a **fixed 20 Hz timestep**: `simTimestep = 0.05 s`.
- `simulatedTime == frameIndex * 0.05` exactly. No frame is ever skipped or
  duplicated; generation does not depend on wall-clock time.
- A 600-frame episode has 600 frames and **599 transitions** (the final step's
  result is discarded). The final frame has no "next" frame.

## 3. The control signal and its alignment (read this first)

The engine has exactly **one** control input per frame: the X coordinate of a
pointer click, in normalized screen space.

**Click → X conversion.** A click at canvas pixel `clientX` (canvas is
`OUTPUT_WIDTH` = 320 px wide) becomes

```
screenX = clamp((clientX - rect.left) / rect.width * 2 - 1,  -1,  1)
```

- `-1` = far left, `+1` = far right, `0` = dead center. The sign convention is
  linear in screen pixels; only the **X** is ever used (Y is ignored).

**Held signal (`controlSignalX`).** The signal is a *held step function*:

- `controlSignalX` equals the X of the most **recent accepted** click, or `0`
  before any click has been accepted.
- It only changes on an **accepted** click. Clicks that fall inside the debounce
  window (0.150 s = 3 frames) are ignored for steering and do **not** change the
  signal.
- The signal **persists across entity boundaries**: when the player walks through
  a doorway into a new entity, the held X is kept and the target is re-resolved
  from it against the new entity's candidates. It does **not** reset to 0.
- The steering **target** is sticky: after an accepted click the target doorway is
  resolved once and held until a new accepted click arrives or the target is no
  longer a live doorway of the active entity (the player walked through it).
- Before the first accepted click the engine is in its "never clicked" default
  mode: the target is re-resolved **every frame** to the reachable candidate
  nearest screen center, and `controlSignalX == 0`. This default mode is
  indistinguishable from "clicked exactly at center" in `controlSignalX` alone —
  which is why `actionHasClicked` exists.

**Alignment rule (the off-by-one that matters).** The per-frame record for frame
`i` is captured *before* the click that drives the step `i → i+1` is applied. So:

> The action for the transition from frame `i` to frame `i+1` is **`actionX[i]`**,
> which equals `controlSignalX[i+1]`. Training must use **`actionX`**, never
> `controlSignalX`, to predict the next frame.

`controlSignalX[i]` is the signal *held at capture time* — it is what a live
client would know at frame `i`, but it is **stale** for predicting frame `i+1`.

## 4. `manifest.json` — episode level

```jsonc
{
  "schemaVersion": 2,
  "episodeId": "<planName>-<split>-<seed>",
  "seed": 12345,
  "split": "train",              // train | val | test
  "profile": "mixture",          // mixture profile name (§9)
  "renderWidth": 1280,           // supersampled render size (OUTPUT * RENDER_SCALE)
  "renderHeight": 960,
  "outputWidth": 320,            // final frame size
  "outputHeight": 240,
  "simTimestep": 0.05,

  // Provenance (identifies exactly what code+config produced this data):
  "configHash":        "sha256 hex of the FULL config object",
  "engineCommit":      "git rev-parse HEAD at generation time ('' if no git)",
  "dirty":             true,     // working tree had uncommitted changes
  "engineSourceHash":  "sha256 of engine/src + engine/config file contents",

  "config": { /* a DISPLAY subset for humans; configHash covers the full config */ },
  "frames": [ /* per-frame records, see §5 */ ],
  "debugFrames": [ /* per-frame debug records, see §6; parallel to frames[] */ ],

  // Clipping failsafe (§7):
  "truncated": false,
  "truncationReason": null,      // e.g. "L2: slide backstop >= 10 consecutive frames"
  "crashFrameIndex": null,       // first frame of the offending event, or null
  "trimmedFrames": 0,            // how many frames were removed from the tail

  "complete": true               // written LAST; a partial episode never has it
}
```

- `complete` is the **completion marker**. It is written last. A directory whose
  `manifest.json` is absent, unreadable, or has `complete !== true` is a *partial*
  episode and is never to be consumed; the runner deletes and redoes it.
- `configHash` hashes the **full** resolved config, not the `config` display
  subset. Use it as the dataset's config fingerprint.

## 5. `manifest.json` — `frames[]` (the training fields)

One record per frame, parallel to `frames/<NNNNNN>.png`.

```jsonc
{
  "frameIndex": 0,
  "simulatedTime": 0.0,          // frameIndex * 0.05
  "imagePath": "frames/000000.png",
  "controlSignalX": 0.0,         // held signal AT capture time (-1..1)
  "actionX": 0.0,                // held signal IN EFFECT during step i -> i+1 (null on final frame)
  "actionHasClicked": false,     // whether >=1 click had been accepted as of step i -> i+1 (null on final)
  "clickX": null,                // RAW click X submitted for step i -> i+1, or null
  "clickAccepted": null,         // true accepted / false debounce-ignored / null no click
  "playerX": 0.5, "playerZ": 1.5,     // debug-only context, NOT fed to the model
  "playerHeading": 1.5708,
  "currentEntityId": "h1",       // id of the entity the player occupies
  "viewUnsafe": false            // camera too close to a rendered wall/jamb (§5b)
}
```

| field | type | meaning |
|---|---|---|
| `frameIndex` | int | 0-based frame number |
| `simulatedTime` | float | `frameIndex * simTimestep` |
| `imagePath` | string | PNG path relative to the episode directory |
| `controlSignalX` | float | held signal **at capture time** (−1..1) |
| `actionX` | float \| null | held signal **in effect during step i→i+1** = `controlSignalX[i+1]`; `null` on the final frame |
| `actionHasClicked` | bool \| null | whether ≥1 click had been accepted as of that step; `false` = "never clicked" default mode; `null` on the final frame |
| `clickX` | float \| null | the RAW click X submitted for the step, or `null` if no click |
| `clickAccepted` | bool \| null | `true` accepted, `false` debounce-ignored, `null` no click submitted |
| `playerX` / `playerZ` / `playerHeading` | float | debug context (not fed to the network) |
| `currentEntityId` | string | active entity id (first is `"h1"`) |
| `viewUnsafe` | bool | camera was too close to a rendered surface at capture time (§5b) |

**Training rule:** feed `actionX` (and `actionHasClicked`) as the action for the
transition from the current frame to the next. `controlSignalX` is kept only for
backward compatibility and for the live-client semantics.

## 5b. `viewUnsafe` — the view-compromise label

The player is a floor point with **no collision radius**, so the camera can end
up arbitrarily close to a wall. Two purely *rendering* artifacts follow, and both
are invisible to the clipping failsafe (§7, which only watches the simulation):

1. **Near-plane clip** — a surface closer than the camera near plane (0.02) is
   clipped away, leaving a hole you can see through.
2. **Single-sided see-through** — walls are drawn front-side only, so a wall
   whose front face points away from the camera is culled; if the camera has
   crossed that wall's plane you look straight through it.

`viewUnsafe` is `true` when, at capture time, either holds: the closest rendered
triangle of the active entity or a maintained neighbour is nearer than
`camera near + viewUnsafeMargin` (default **0.10** world units, recorded in
`manifest.config.viewUnsafeMargin`), **or** the camera sits behind a single-sided
wall plane with geometry in front of it.

- It is a **label only**. It never aborts, trims, or otherwise changes an
  episode — unlike the clipping failsafe (§7), which does stop episodes.
- **Loader rule:** a training window (frame → action target) must be **dropped**
  if any frame it covers, *including its action target frame*, is flagged. The
  frames stay on disk and stay in the manifest; the loader filters them out.
- Occurrence is rare (< 1% of frames in the pilot); see the diagnostics report
  (§11 of `report.md`).

## 6. `manifest.json` — `debugFrames[]` (separable debug section)

`debugFrames[i]` is parallel to `frames[i]`. A training loader should ignore this
whole section cheaply (it can be dropped without affecting training).

```jsonc
{
  "frameIndex": 0,
  "playerOmega": 0.0,            // angular velocity, rad/s
  "lastClickedX": null,          // raw last accepted click X (null before any click)
  "targetDoorwayId": null,       // doorway currently steered toward
  "targetPointX": null, "targetPointZ": null,   // steering target point (world space)
  "targetPhase": "approach",     // "approach" | "through"
  "collisionEvents": 0,          // cumulative count of wall-slide redirections
  "wallOverlap": false,          // player ever ended up inside a wall/void
  "candidates": [                // this frame's doorway candidates
    { "doorwayId": "d2", "screenX": -0.31, "reachable": true, "screenDistance": 0.31 }
  ],
  "persona": "decisive",         // persona active when this step's click was decided (§9)
  "intentDoorwayId": null,       // doorway the bot aimed at, or null
  "slid": false                  // wall-slide backstop engaged during step i -> i+1
}
```

- `collisionEvents` is **cumulative** across the episode (a running counter), not
  per-frame.
- `slid` is **per-step** and aligned with `actionX` (it reports the step i→i+1).

## 7. Clipping failsafe and `aborted.jsonl`

The simulator occasionally drives the agent into a wall. The failsafe keeps those
frames out of the dataset; it is a safety net, not a fix, and only affects
headless agent episodes (human play is unaffected).

Three levels, evaluated per frame from engine state (never image analysis):

- **L1 (hard clip)** — `wallOverlap`: the player is inside wall geometry or off the
  walkable floor. **Triggers.**
- **L2 (sustained contact)** — the wall-slide backstop engaged for `CLIP_L2_RUN_LENGTH`
  (10) consecutive frames. **Triggers.**
- **L3 (brief graze)** — the backstop engaged for < 10 frames while the camera
  stayed clear. **Not a crash**; counted and reported only.

When L1/L2 fires, let `c` be the first frame of the offending event and
`T = CLIP_TRIM_SECONDS` (8 s = 160 frames):

1. Stop the episode immediately.
2. Delete every frame from `c − T` onward **from disk** (the PNGs are physically
   removed, not just omitted). Recompute the action-aligned fields so the new
   final frame has `actionX = null`, then write the manifest with
   `truncated: true`, `truncationReason`, `crashFrameIndex = c`, and
   `trimmedFrames` (the number removed).
3. If fewer than `MIN_KEPT_FRAMES` (60) remain, delete the **whole** episode
   directory instead.
4. Append one JSON line to `<planDir>/aborted.jsonl`:

```jsonc
{
  "episodeId": "...", "seed": 6, "split": "train",
  "persona": "spammer",            // persona active at the crash
  "level": "L2",                   // "L1" | "L2"
  "reason": "slide backstop >= 10 consecutive frames",
  "crashFrameIndex": 375,
  "framesKept": 215,               // 0 when the whole dir was deleted
  "clickSequence": [ { "frameIndex": 0, "clickX": 1.0 }, /* ... */ ],
  "stateDump": [ { "frameIndex": 355, "x": ..., "z": ..., "heading": ...,
                   "entityId": "h1", "targetDoorwayId": "d3" }, /* last ~20 */ ]
}
```

An aborted episode counts as **done** for resume purposes; it is never retried or
replaced automatically. At the end of a plan run the runner prints the aborted
count/share and the L3 graze rate, and warns/exits non-zero if the share exceeds
`CLIP_ALERT_SHARE` (5%, after ≥20 episodes), stopping early past `CLIP_STOP_SHARE`
(25%).

## 8. Run plans

A plan is a small JSON file. The CLI can generate one (`--make-plan`) and run one
resumably (`--plan`):

```jsonc
{
  "name": "pilot_v1",
  "profile": "mixture",            // mixture profile (§9)
  "framesPerEpisode": 600,
  "outputRoot": "/abs/path/engine/data",
  "seedRanges": {                  // DISJOINT ranges per split
    "train": { "base": 0,          "count": 120 },
    "val":   { "base": 1000000,    "count": 15 },
    "test":  { "base": 2000000,    "count": 15 }
  },
  "episodes": [
    { "seed": 0, "split": "train", "profile": "mixture" },
    /* ... */
  ]
}
```

## 9. Mixture profiles and personas

Each episode runs under a **mixture profile**; the profile is recorded in the
manifest's `profile` field. The bot that generates clicks is one of seven
**personas**, drawn per ~100–300-frame segment from the profile's weights. The
active persona for each step's click is recorded in `debugFrames[].persona`.

Profiles (weights need not sum to 1):

- `mixture` (default): idle 10, decisive 30, late 10, dither 10, prober 15,
  spammer 5, uniform 20.
- `uniformOnly`: uniform 1 (the legacy placeholder behavior).
- `doorAwareOnly`: decisive 3, late 1, dither 1.

Personas: `idle` (no clicks; stale-held-X regime), `decisive` (uniform reachable
door + Gaussian jitter after a reaction delay), `late` (long delay → marginal
reachability/fallback), `dither` (re-click a different door every 10–40 frames),
`prober` (plateau boundaries / extremes / center / walls), `spammer` (2–8 click
bursts), `uniform` (U(−1,1) every 20–80 frames). Bot randomness comes from a
seeded generator derived from the episode seed and is entirely separate from the
world generator, so bot choices never perturb world generation.

## 10. Determinism

For a given plan + seed, a full episode is byte-for-byte reproducible on the same
machine/GPU (two runs give byte-identical PNGs and identical manifests). The
world content is a function of `(seed, path)` — the order in which entities are
lazily generated — and the only nondeterminism in the pipeline was the random
`episodeId`, which is now deterministic (`<planName>-<split>-<seed>`).
