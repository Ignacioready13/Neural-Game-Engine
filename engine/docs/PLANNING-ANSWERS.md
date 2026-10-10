# PLANNING ANSWERS — NeuralGameEngine (READ-ONLY INVESTIGATION)

Investigation date: 2026-10-08. Engine HEAD: `105f8e4` ("engine stability improvements").
Everything below is verified against the code as it exists today, with file + line citations,
or marked **RUN** with the exact command + real output, or marked **UNVERIFIED**/**NOT RUN**.
Where README/CHANGES-LOG disagree with code, the code wins and the disagreement is flagged.

---

## SECTION 1 — THE CONTROL SIGNAL AND CLICK SEMANTICS

### 1.1 [P1] Click end-to-end — is it "click once to set a target" or hold/drag? Which event?

**Direct answer:** It is **click-once-to-set-a-target**. Only the pointer **`click`** event is used
(no `mousedown`/`mouseup`, no drag/hold tracking). The click's **X is the only input**; Y is discarded.

**Evidence — human/recorder path:**
- `engine/src/app/main-interactive.ts:26-27` — `canvas.addEventListener('click', (ev) => { engine.submitClick(clientXToScreen(ev.clientX)); })`.
- `engine/src/app/main-record.ts:39-41` — same `click` listener: `canvas.addEventListener('click', (ev) => { pendingClickX = clientXToScreen(ev.clientX); })`.
- `clientXToScreen` (main-interactive.ts:20-24, main-record.ts:30-34) converts a client pixel X to `[-1,1]` and drops Y entirely.

**Evidence — headless/bot path:**
- `engine/src/headless/driver.ts:178-203` — per frame: `clickX = bot(frameIndex)`, then `step(clickX)`.
- `engine/src/app/main-headless.ts:93-97` — `step(clickX)` does `if (clickX !== null) e.submitClick(clickX); e.step();`.
- `engine/src/sim/Engine.ts:86-89` (`submitClick`) stores `pendingClickX`.
- `engine/src/sim/Engine.ts:96-100` — on the next `step()`, `steering.handleClick(pendingClickX, simulatedTime)`.

**Every function the click passes through until it changes simulation state:**
`click event → clientXToScreen → Engine.submitClick → (next step) Steering.handleClick → Steering.update → resolveTarget (doorwayResolution) → Steering.step`. State changed: `_lastClickedX` (held signal), `_pendingResolution`, `_targetDoorwayId` (sticky target), then `player.heading/angularVelocity/x/z` via `Steering.step`.

**Caveat:** none — there is no hold/drag state anywhere.

### 1.2 [P1] What exactly is controlSignalX?

**Direct answer:** **(b)** the X of the most recent **ACCEPTED** click (i.e. not discarded by debounce),
normalized to `[-1,1]`; `0` is the screen-center default **before the first accepted click**. It is a
held step-function: it changes only when a click is accepted, and stays constant otherwise. It is **not**
derived from the chosen doorway.

**Evidence:**
- `engine/src/control/steering.ts:50-52` — `get lastClickedX()` is `_lastClickedX`, set only on an accepted click.
- `engine/src/control/steering.ts:79-84` — `accepted = simTime - _lastAcceptedTime >= debounceSeconds`; only if `accepted` does it set `_lastClickedX = x`.
- `engine/src/sim/Engine.ts:211` — `getState()` returns `controlSignalX: this.steering.lastClickedX ?? 0`.
- `engine/src/headless/driver.ts:194` — the manifest frame records `controlSignalX: state.controlSignalX`.

### 1.3 [P1] Debounce — what happens to target, signal, pulse; are ignored clicks logged?

**Direct answer:** A click arriving inside the debounce window (`DEBOUNCE_SECONDS = 0.150`):
(i) the **steering target is unchanged** (no re-resolution), (ii) **controlSignalX is unchanged**
(`_lastClickedX` not updated), (iii) the **click pulse still fires** (visual only). Ignored clicks are
**NOT recorded in the headless manifest** (it only has `controlSignalX`, which didn't change) but **ARE
recoverable from the playtest manifest** via its `clickedX` field.

**Evidence:**
- `engine/src/control/steering.ts:74-86` — every click updates `_lastClickTime` and `_lastClickScreenX` (pulse), but only `accepted` clicks update `_lastClickedX`/`_lastAcceptedTime`/`_pendingResolution`.
- `engine/config/config.ts:144` — `DEBOUNCE_SECONDS = 0.150`.
- Headless manifest frame schema `engine/src/core/types.ts:200-212` — has only `controlSignalX`, no raw-click field.
- Playtest frame has `clickedX` (raw click) + `lastClickedX` (accepted value): `engine/src/app/main-record.ts:99,54`.

### 1.4 [P1] Is the target STICKY or re-resolved every frame?

**Direct answer:** **STICKY**, but with a precise nuance that depends on click state:

- **After an accepted click:** the target is resolved **once** (on the next `update()`), then **held** until
  (a) a new accepted click arrives, or (b) the target doorway is **no longer a live doorway of the active entity**
  (i.e. the player walked through it into the neighbor). When (b) happens, it is re-resolved from the **held** click X.
- **With no click ever (`lastClickedX === null`):** the target is **re-resolved every frame** to the reachable
  doorway nearest screen center (see 1.6).

**Evidence:**
- `engine/src/control/steering.ts:108-141` (`update`) — `_pendingResolution` resolves once (113-121); idle null-click re-resolves every frame (123-127); held-click case holds the sticky target while `liveDoorwayIds.includes(targetDoorwayId)` (131-136), else re-resolves (138-140).
- The stickiness rationale is documented in `steering.ts:26-32` and `Engine.ts` passes `liveDoorwayIds` = **all** doorways of the active entity (not just projectable ones): `engine/src/sim/Engine.ts:106`.

### 1.5 [P1] What happens to the previous click when the player enters a NEW entity?

**Direct answer:** **The held X persists** and **re-resolves against the new entity's candidates**; it does
**not** reset to 0 or screen-center. `controlSignalX` in the manifest does **not** change at that moment
(it still equals the last accepted click). Steering continues to be influenced by the held X in the new entity.

**Evidence:**
- `engine/src/control/steering.ts:138-140` — when the sticky target is no longer a live doorway (player left the old entity), `resolveTarget(candidates, this._lastClickedX, null)` re-resolves using the **held** `_lastClickedX`, now against the new entity's doorways.
- `Engine.step` passes the new entity's `liveDoorwayIds` and candidates after the transition: `engine/src/sim/Engine.ts:103-122`.

**Where logged signal and steering can disagree:** the signal only changes on an accepted click; the steering
*target* can change (re-resolve on entity entry, or target drops to `null` if no candidate is reachable) without
the signal changing. So a single `controlSignalX` value can correspond to different target doorways at different
times — the signal under-specifies steering.

### 1.6 [P1] Default behavior (no active click): which doorway, and per-frame or on-entry?

**Direct answer:** The **reachable doorway nearest screen center** (`referenceX = 0`), re-evaluated **every
frame** while `lastClickedX === null`. This is "straightest on screen", not "straightest in world heading".

**Evidence:**
- `engine/src/control/steering.ts:123-127` — `if (this._lastClickedX === null) { resolveTarget(candidates, null, ...); }` every frame.
- `engine/src/control/doorwayResolution.ts:27` — `referenceX = clickX ?? 0`.

### 1.7 [P1] Coordinate convention — pixel→normalized mapping, sign, linearity, supersampling, doorway screen X

**Direct answer:** `screenX = clamp((clientX - rect.left) / rect.width * 2 - 1, -1, 1)`. **Left is negative (−1),
right is positive (+1).** The mapping is **linear in screen pixels**. Supersampling (`RENDER_SCALE`) does **not**
affect it: the input uses the **canvas** bounding rect, and the canvas is `OUTPUT_WIDTH` (320), not the 1280 supersampled
target. Aspect ratio affects it only indirectly (through the projection, not the click mapping).

Doorway on-screen X is computed by the **same camera / same FOV / same projection**: full perspective projection
of the doorway's approach point to normalized device coordinates.

**Evidence:**
- `engine/src/app/main-interactive.ts:20-24` — the linear pixel→`[-1,1]` formula and `clamp`.
- `engine/src/app/main-headless.ts:23-30` — canvas is `CONFIG.outputWidth × outputHeight` (320×240).
- `engine/src/render/scene.ts:95-116` — `positionCamera` then `projectToScreenX(wx,wz,wy)`: `v.applyMatrix4(matrixWorldInverse)`; if `v.z >= -1e-9` return null; `v.applyMatrix4(projectionMatrix)` (divides by w); return `v.x`.
- `engine/src/render/scene.ts:29-33` — `verticalFovDeg(horizontalFovDeg, aspect)` derives vertical FOV from horizontal 85° and aspect.
- `engine/src/sim/Engine.ts:249-274` (`collectCandidates`) projects each doorway's **approach point** via `renderer.projectToScreenX`.

### 1.8 [P1] Range / resolution; can it be ±1 and exactly 0; distinguish "never clicked" vs "center"?

**Direct answer:** Range is `[-1, 1]` inclusive. It **can** be exactly `-1` or `+1` (clamped). It **can** be exactly
`0` after a real click (a click at exact screen center). In the **headless** manifest there is **no way to distinguish
"never clicked" from "clicked dead center"** — both are `0`. The **playtest** manifest can distinguish them: it records
`lastClickedX` (`null` vs `0`) separately.

**Evidence:**
- `steering.ts:75` — `clamp(screenX, -1, 1)`; `steering.ts:50-52` — `_lastClickedX` null before any click.
- `Engine.ts:211` — `controlSignalX: this.steering.lastClickedX ?? 0` (null → 0).
- Playtest `main-record.ts:91` records `lastClickedX: state.lastClickedX ?? null`.

**Practical resolution:** the placeholder bot's clicks are `rng()*2 - 1` (uniform in `(-1,1)`), so it essentially
never produces exactly ±1 or exactly 0 (bot.ts:30). Real human clicks are continuous pixel positions, so any value in
`[-1,1]` is possible.

### 1.9 [P2] Other reasons clicks are ignored?

**Direct answer:** Only **debounce** causes a click to not change the steering target/signal. There is **no** ignore
during a gated turn phase, during entity transitions, or when no candidate is reachable. (A click with no reachable
candidate still sets `_lastClickedX` and the pulse; the target resolves to `null`/previous, but the signal changes.)

**Evidence:** `steering.ts:74-86` is the only gate; `resolveTarget` handles the no-reachable case by returning null/previous (`doorwayResolution.ts:55-62`) without affecting `_lastClickedX`.

### 1.10 [P2] Any other input besides click X?

**Direct answer:** **No.** Keyboard only ends a recording (`Q`/`Escape` in main-record.ts:130-134) and never affects
simulation. Scroll and Y are ignored. The only per-frame simulation input is the click X.

---

## SECTION 2 — FRAME / SIGNAL TIMING ALIGNMENT

### 2.1 [P1] Exact per-frame order of operations

**Headless loop** (`engine/src/headless/driver.ts:178-203`) — **capture-before-step**:

```
for frameIndex in 0..N-1:
  clickX = bot(frameIndex)                 # read input for THIS frame
  capture = page.evaluate(captureFrame)    # render current state + grab PNG
  state   = page.evaluate(getState)        # read controlSignalX/player BEFORE step
  writePng(frames/frameIndex.png)          # disk write (Node)
  push FrameRecord{ controlSignalX: state.controlSignalX, player: state.player, ... }
  page.evaluate(step(clickX))              # apply click + advance one timestep
```

`Engine.step()` internal order (`engine/src/sim/Engine.ts:92-168`):
1. process pending click → `steering.handleClick` (line 96-100)
2. collect candidates (project + reachability) (103-105)
3. `steering.update` → resolve target (109)
4. find doorway, `doorwayPhase`, `doorwayTargetPoint` (115-122)
5. `steering.step` (heading/omega/position) then `resolveWallCollision` (126-129)
6. entity-tracking with hysteresis (137-144)
7. wall-overlap check (149-159)
8. `updateFlicker`, `simulatedTime += dt`, `frameIndex += 1`, pulse aging (161-167)

**Human recording loop** (`engine/src/app/main-record.ts:74-103`) — **step-then-record**:

```
stepAndRecord():
  clickX = pendingClickX
  if clickX !== null: engine.submitClick(clickX)   # queue
  engine.step()                                     # apply + advance
  state = engine.getState(); capture = engine.captureFrame()
  push RecordFrame{ clickedX: clickX, controlSignalX, player, ... }
  pendingClickX = null
```

### 2.2 [P1] Which frame does a click first affect? Which manifest index records the new signal?

**Direct answer (plain statement):**
> **The transition from frame i to frame i+1 is driven by `controlSignalX[i+1]`, NOT by `controlSignalX[i]`.**

A click submitted at step i first affects **frame i+1** (both its image and its `controlSignalX`).
`controlSignalX[i]` is the value held *before* step i — i.e. it is **stale** with respect to the i→i+1 change.

**Proof (RUN), renderer-free replica of `Engine.step`** (`data/scratch/timing.mts`, Part A; a click `screenX=0.6` submitted at step 5):

```
frame | controlSignalX | heading
  0   |   0.000 | 1.5708
  1   |   0.000 | 1.5558
  2   |   0.000 | 1.5362
  3   |   0.000 | 1.5193
  4   |   0.000 | 1.5047
  5   |   0.000 | 1.4922     <- click submitted in step 5
  6   |   0.600 | 1.4946     <- first frame where signal AND heading direction change
  7   |   0.600 | 1.5121
  8   |   0.600 | 1.5446
```

The heading reversed direction (was decreasing 1.5708→1.4922, then increasing after the click) only at frame 6,
confirming both the signal and the motion first change at frame 6.

**Evidence:** `driver.ts:174-203` (comment "capture-before-step" + loop order); `Engine.ts:96-100` (click processed at step start).

### 2.3 [P1] RUN: click-to-motion latency

**Result (RUN), 2,808 reachable-doorway samples over 8 seeds** (`data/scratch/timing.mts`, Part B):

| Metric | min | median | max | never (within 60) |
|---|---|---|---|---|
| frames to >1° heading change | 1 | 2 | 60 | 388 |
| frames to aligned (<1° error) | 1 | 10 | 60 | 296 |

Target distances sampled: min 1.0, median 8.9, max 28.8 world units.

**Interpretation:** the player starts turning within 1–2 frames (one jerk-limited `dt=0.05s` step) of an accepted
click; full alignment to the doorway typically takes ~10 frames (0.5 s). The ~388 samples that never reached >1°
within 60 frames are clicks on a doorway nearly straight ahead (heading error already <1°, so the click changes
little); the ~296 that never "aligned" are targets the player passed/left mid-turn. This variance is inherent to
jerk limiting (`JERK_THRESHOLD=6.0 rad/s²`) + `MAX_TURN_RATE=2.5` + the two-phase gate radii.

### 2.4 [P2] Duplicated/skipped frames? simulatedTime == frameIndex·timestep?

**Direct answer:** No duplicated or skipped frames in the headless pipeline. `simulatedTime === frameIndex * simTimestep`
exactly for every manifest frame (frame i → i·0.05). Frame 0 is the initial state (before any step). The final
`step(N-1)` result is discarded (never captured), so the manifest has exactly N frames at t=0 … (N−1)·dt.

**Evidence:** `Engine.ts:162-163` (`simulatedTime += dt; frameIndex += 1` always together); `driver.ts:178-203`.
**Caveat:** the **human recorder** is 1-based: its first frame has `frameIndex=1` (step-then-record), so its frames are t=1·dt … N·dt (see playtest manifest frame[0].frameIndex=1, verified in §9).

---

## SECTION 3 — STEERING AND REACHABILITY

### 3.1 [P1] Candidate-collection logic

**Direct answer:** A candidate is any **doorway of the active entity** whose **approach point** projects to a screen
X in front of the camera (projection returns null if behind). There is no "enterable" gate beyond reachability. Each
candidate carries: `doorwayId`, `screenX` (NDC, may be slightly outside `[-1,1]`), `reachable` (bool), `screenDistance`
(= |screenX − referenceX|, where reference = last click X or 0). The count is variable = number of active-entity
doorways (no fixed cap); observed up to 6+ (histogram in §7.2).

**Evidence:**
- `engine/src/sim/Engine.ts:249-274` (`collectCandidates`).
- `engine/src/core/types.ts:162-170` (`CandidateDoorway`).
- `engine/src/sim/navigation.ts:64-66` — candidates use the **approach point** (corner for a bend, opening center otherwise).

### 3.2 [P1] Exact click-resolution algorithm

**Direct answer** (`engine/src/control/doorwayResolution.ts:20-63`):
1. `referenceX = clickX ?? 0`.
2. Stable-sort candidates by `|screenX − referenceX|` ascending (tie-break by original index).
3. If the nearest is `reachable`, it wins.
4. Else search **outward in increasing screen distance, biased same-side-first**: split the rest into the same
   side of `referenceX` as the nearest candidate and the other side; return the first reachable in the same side,
   then the first reachable in the other side.
5. If none reachable, return the **previous target if still a live candidate**, else `null` (steer straight).

### 3.3 [P1] What does "reachable" depend on? Late-click differences (two examples)

**Direct answer** (`steering.ts:196-216` + `requiredArcLength` 223-248): reachable ⇔
(a) the approach point is in the **front cone** (|angle| ≤ `REACHABILITY_MAX_ANGLE_DEG`=75°), **and**
(b) the straight-line distance ≥ the **jerk-limited minimum-time arc length** to finish the turn (a conservative
over-estimate that accounts for current `angularVelocity`). It does **not** depend on gate state or phase.

**Two concrete late-click examples:**
1. **A bend doorway**: candidate is tested against its **approach point = the corner** (not its `entryPoint`, a waypoint `TURN_WAYPOINT_DEPTH`=3 units *past* the corner). Clicking it early is fine; clicking it *after* the player is abreast of the corner changes nothing about reachability (still in cone) but the click may resolve to a *different* doorway if another is now nearer by screen X.
2. **A side doorway clicked early vs 1.5 units before it**: reachability is a function of current heading, angular velocity and distance. Close to the opening the player may be turning such that `requiredArcLength` exceeds the remaining distance (or the doorway leaves the 75° cone as it goes behind), so the **same** click X can be accepted (reachable) when clicked early but rejected (falls back to another doorway) when clicked late. This is the "same input → different outcome depending on timing" behavior.

### 3.4 [P1] Retarget mid-turn? Phases that disallow it?

**Direct answer:** **Yes**, retarget is allowed at any time (including mid-turn), gated only by debounce (0.150 s).
Heading continuity is preserved because `Steering.step` carries `angularVelocity` across calls and jerk-limits the
rate change. There is **no** phase that disallows or delays retargeting.

**Evidence:** `steering.ts:149-178` (`step` uses `player.angularVelocity` + jerk limit; desired heading computed from current target each frame); `steering.ts:74-86` (only debounce gates).

### 3.5 [P2] Is walking speed constant?

**Direct answer:** **Yes.** `WALK_SPEED = 4.0` units/s, applied every step regardless of turn/phase; the player never stops.

**Evidence:** `engine/config/config.ts:24`; `steering.ts:174-175` (position advances by `walkSpeed * dt` unconditionally).

### 3.6 [P2] Can the player get stuck / pinned / no target?

**Direct answer:** By design no, and the RUN stats confirm it. Across **480,000 frames (200 seeds × 1200 × 2 bots)**:
**0 pinned frames, 0 wall-overlap frames**. Wall **contact** (collision slide) does occur (~45–51 per 1,000 frames,
worst episode 1,137) but never stalls — the collision backstop slides the player along the wall and the heading keeps
turning. When there is genuinely no target (`null`), `steering.step` keeps heading straight (`desiredHeading = player.heading`,
steering.ts:150-153) and `controlSignalX` is unchanged.

**Evidence:** `collision.ts:18-37`; `steering.ts:149-153`; §7.5 RUN numbers.

---

## SECTION 4 — BOT INTERFACE AND HEADLESS ARCHITECTURE

### 4.1 [P1] Placeholder bot code path; Node or in-page?

**Direct answer:** The bot runs **in Node**, not in the page. It is a pure function `(frameIndex) => number | null`
that returns a deterministic pseudo-random click X (uniform in `[-1,1)`) every 20–80 frames, `null` otherwise. It is
fed across the Puppeteer bridge via `page.evaluate(step(clickX))`.

**Evidence:**
- `engine/src/headless/bot.ts:20-39` (`makePlaceholderBot`): seeded `createRng(seed)`, `nextClickAt = intRange(rng,20,80)`, click value `rng()*2 - 1` (line 30).
- `engine/src/headless/driver.ts:171,179,202` — `const bot = makePlaceholderBot(seed)` in Node; `step(clickX)` via `page.evaluate`.

### 4.2 [P1] What a smarter bot could read; is it exposed?

**Direct answer:** The following is **already exposed** through `getState()` (`engine/src/sim/Engine.ts:181-225`):
`player {x,z,heading,angularVelocity}`, `activeEntityId`, `frameIndex`, `simulatedTime`, `controlSignalX`, `wallOverlap`,
`collisionEvents`, `lastClickedX`, `targetDoorwayId`, `targetPoint {x,z}`, `targetPhase`, and **`candidates[]`**
(each `{doorwayId, screenX, reachable, screenDistance}`).

**Not exposed / needs plumbing:** per-entity type/size of the *active* entity is recoverable indirectly (only
`activeEntityId`, not its kind/termination), and "distance to next doorway" / "time since entity entry" are not
direct fields (would need to be computed from `targetPoint` + `simulatedTime`, or added). Plumbing point: extend
`getState()` in `Engine.ts` and the `BackroomsState`/`api.ts` types, and re-run the typecheck. Note the bot runs in
Node, so a smarter bot would add a `getState()` read per frame (extra bridge round-trip) unless it is moved in-page (see 4.3).

### 4.3 [P1] Per-frame bridge cost; in-page bot feasibility

**Direct answer:** Yes — the pipeline crosses the bridge **three times per frame** (`captureFrame`, `getState`, `step`),
plus a Node-side `writePng`. Moving a bot in-page is straightforward: `main-headless.ts` already exposes `window.__backrooms`
with `step/getState/captureFrame`; an in-page bot would just call `window.__backrooms.getState()` directly instead of
round-tripping through Node. The only reason the placeholder bot is in Node is simplicity.

**Evidence:** `driver.ts:182,185,202` (three `page.evaluate` calls); `main-headless.ts:88-110`.

### 4.4 [P2] Seed assignment; explicit seed list + policy name?

**Direct answer:** The CLI takes a single `--seed` base and derives per-episode seeds as `baseSeed + i`
(worker index). Episode **ids** are `randomUUID()` (non-deterministic). There is **no** support for an explicit list of
seeds or a per-episode policy name today — that would require new CLI/plumbing.

**Evidence:** CLI arg parsing in `engine/src/headless/cli.ts` (summary: `--concurrency/--episodes/--frames/--seed/--output`); `driver.ts:141` `episodeId = opts.episodeId ?? randomUUID()`; `config.ts:226` `EPISODE_SEED: null` (random per episode when unset).

### 4.5 [P2] Config overrides without editing config.ts?

**Direct answer:** **No.** The CLI overrides only episode count / frames / seed / output dir / concurrency / serve-port /
build. `OUTPUT_WIDTH`/`OUTPUT_HEIGHT` (and all other `CONFIG` fields) are compile-time constants read from `config.ts`
and cannot be overridden per-run. Changing resolution requires editing `config.ts` and rebuilding `dist/`.

**Evidence:** `config.ts:297-347` (`CONFIG` is frozen); `cli.ts` flags (summary); `main-headless.ts:23-30` reads `CONFIG.outputWidth/Height`.

---

## SECTION 5 — DETERMINISM AND REPLAY

### 5.1 [P1] Is world content f(seed) alone, or f(seed, path)?

**Direct answer:** **f(seed, path)** — **not** seed alone. The `World` owns a **single shared seeded PRNG**, and
generation is **lazy**: `maintain(activeId)` generates all not-yet-generated neighbors of the active entity (in doorway
order) and disposes everything else. So an entity's contents are determined by the RNG state at the moment its generator
runs, which is determined by the full history of `maintain()` calls — i.e. **the path the player takes**. Two different
click sequences from the same seed reach entities in a different order (or different entities) and therefore consume RNG
differently, producing **different** entity contents for "the same" neighbor. Within a single episode (one path) everything
is fully deterministic (proven byte-identical in 5.2).

**Evidence:**
- `engine/src/world/World.ts:25-36` — `this.rng = createRng(seed)` (single PRNG); `maintain` 60-106 generates neighbors in doorway order (65-73).
- Re-disposal/regeneration (88-104): a disposed neighbor's doorway is reset to ungenerated (`targetGenerated=false` except `side==='start'`), so a **re-entered** entity regenerates its neighbor with **advanced RNG state → different contents** than the first time.

### 5.2 [P1] RUN: replay a scripted episode twice — byte-identical?

**Result (RUN):** Two headless episodes, same seed `424242`, 600 frames each, concurrency 1:

```
GPU renderer: ANGLE (NVIDIA, Vulkan 1.4.329 ... NVIDIA GeForce RTX 5070 Laptop GPU ...)
byte-identical PNGs: 600 / 600  (0 mismatched)
frame-state mismatched: 0 / 600  (playerX/Z/heading/controlSignalX)
max |playerX/Z| delta: 0   max |heading| delta: 0
```

Only `episodeId` differs (it's a `randomUUID`). **PNGs are byte-identical and position/heading are bit-identical.**

**Command:** `node node_modules/tsx/dist/cli.mjs src/headless/cli.ts --episodes 1 --frames 600 --seed 424242 --output <dir> --concurrency 1` (twice), then SHA-256 per PNG + field diff.

### 5.3 [P1] RUN: replay the HEADFUL recording through the same logic — do states match?

**Result (RUN):** Replayed the playtest (`seed 3969209238`, 578 frames) by feeding its recorded per-frame `clickedX`
through a faithful renderer-free replica of `Engine.step` (the same logic the regression tests validate), then compared
player state to the recorded state:

```
clicks replayed: 2
state mismatch (tol 1e-9 or entity id): 0 / 578
max |dx| = 8.88e-16   max |dz| = 0   max |heading| = 3.33e-16
entity-id mismatches: 0
```

The simulation state is **bit-identical** (deltas at machine epsilon). This is expected: both modes run the **same
shared `Engine`**; only the input source and GPU launch args differ.

**Rendering caveat (the "frames match" half):** the headful recorder launches Chrome with `HEADFUL_LAUNCH_ARGS`
(`--use-angle=gl`, Intel iGPU/Mesa — `driver.ts:98-104`) while headless uses `GPU_LAUNCH_ARGS` (`--use-angle=vulkan`,
NVIDIA — `driver.ts:69-76`), and the headless CLI hardcodes `makePlaceholderBot` (no custom-click injection path), so I
did **not** byte-compare headful vs headless **PNGs** — they will differ by rasterizer even though the simulation state
is identical. The state match above is the meaningful determinism result.

### 5.4 [P2] Sources of nondeterminism

**Identified:**
- `randomUUID()` for `episodeId` (manifest-level, not frame-level) — `driver.ts:141`.
- `Math.random` — **not used** in world/steering/bot (all seeded `mulberry32` via `createRng`). `Date.now`/`performance.now` are used only for **wall-clock timing** (flicker uses simulated time; the recorder's `performance.now` accumulator — main-record.ts:71-73 — drives *when* steps fire but each step is still fixed-dt and identical).
- **Flicker is off** (`FLICKER_ENABLED=false`), so no time-varying lighting.
- Texture generation is procedural/Canvas and deterministic given the seed (no network, no `Date`).
- Floating point: deterministic given identical code/GPU; the **rendered pixels** depend on the GPU/rasterizer (Intel vs NVIDIA differ), but **simulation state** does not.
- The **path-dependence** in 5.1 is the dominant "nondeterminism" to watch for cross-bot/cross-policy comparability.

---

## SECTION 6 — DATASET FORMAT, SIZE, AND THROUGHPUT

### 6.1 [P1] RUN: one 600-frame episode

**Result (RUN, seed 424242, 600 frames):**

- Per-frame record fields (headless) — one entry:

```json
{ "frameIndex": 0, "simulatedTime": 0, "imagePath": "frames/000000.png",
  "controlSignalX": 0, "playerX": 1.5, "playerZ": 0, "playerHeading": 0, "currentEntityId": "h1" }
```

  (Note `currentEntityId` starts at **`h1`**, not `h0` as the README example shows — doc drift.)
- PNG sizes: mean **80,530 B** (min 47,149, max 98,881, p50 84,492, p90 91,506).
- `manifest.json`: **184,682 B**.
- Total episode dir: **48 MB**.

Extrapolation (using mean PNG + per-frame manifest overhead ≈ 308 B/frame):

| frames | approx total |
|---|---|
| 100,000 | ~8.1 GB |
| 500,000 | ~40.4 GB |
| 1,000,000 | ~80.8 GB |

### 6.2 [P1] RUN: throughput benchmark

**Result (RUN), 8 episodes × 600 frames per concurrency level** (`data/scratch/bench.sh` via `src/headless/cli.ts`):

| concurrency | elapsed | aggregate fps | per-instance fps | peak GPU mem |
|---|---|---|---|---|
| 1 | 42.41 s | 113.2 | 113.2 | 103 MiB |
| 2 | 22.58 s | 212.6 | 106.3 | 177 MiB |
| 4 | 12.83 s | 374.0 | 93.5 | 334 MiB |
| 6 | 12.70 s | 377.8 | 63.0 | 498 MiB |
| 8 | 9.17 s | 523.3 | 65.4 | 654 MiB |

**Best concurrency:** 8 gives the highest aggregate throughput (523 fps); 4 gives the best per-instance throughput with
more headroom (GPU mem scales ~82–84 MiB/instance). CPU use was **not sampled** in this benchmark (UNVERIFIED).

**Hours to generate 500,000 frames:** at concurrency 8 ≈ **0.27 h (≈16 min)** (500000/523.3 ≈ 955 s); at concurrency 4 ≈ **0.37 h (≈22 min)**.

### 6.3 [P1] Per-frame time budget — **PARTIAL / NOT COMPLETED**

**What is known (from code + total):** at concurrency 1, ~**8.8 ms/frame** total (113 fps). The per-frame pipeline stages
(in `Engine.captureFrame` → `Renderer.render` → `resolution.ts`) are:

1. **Render scene** to supersampled target (1280×960, `RENDER_SCALE=4`).
2. **Box-filter downsample** GPU pass → 320×240.
3. **Blit** to output canvas.
4. **`gl.readPixels`** GPU→CPU readback (rows flipped).
5. **`putImageData`** + **`canvas.toDataURL('image/png')`** (browser CPU PNG encode).
6. **Bridge** serialization of the data URL (included in the `captureFrame` evaluate).
7. **Node `writePng`**: base64 decode + `fs.writeFile` (per frame).

**Fine-grained numeric split was NOT completed** (investigation was cut short). I did not instrument the in-page
render/downsample/readback/encode stages separately, and I did not leave instrumentation in source per the ground rules.
**To finish:** add timing inside `Renderer.render`/`captureFrame` (scratch only) or use CDP tracing. The dominant cost is
expected to be the PNG encode + readback (the fixed-timestep sim itself is negligible).

### 6.4 [P1] Manifest written incrementally? Crash/resume behavior?

**Direct answer:** The manifest is written **once at the end** (not incrementally); PNGs are written **per frame**. A crash
at frame 500/600 leaves the `frames/` dir with 500 PNGs but **no `manifest.json`** (or a missing/partial one), so a partial
episode is distinguishable by the **absence of a manifest** (and its frame count not matching the PNG count). The worker
pool does **not** resume — a batch restart re-runs everything. There is no completion marker beyond the presence of the
manifest.

**Evidence:** `driver.ts:205-233` (builds `frames[]` in memory, writes `manifest.json` at the end via `writeManifest`); `writePng` called per-frame at `driver.ts:188`.

### 6.5 [P2] Long-episode memory growth — **NOT RUN**

I did **not** run a 10,000-frame episode to measure page/Node memory growth (investigation cut short). **UNVERIFIED.**
What the code implies: `World.maintain` disposes non-neighbor entities (geometry + materials) each step (`World.ts:60-106`,
`disposeEntity` 161-180), so the *live* entity set is bounded to active + direct neighbors; however the **manifest `frames[]`
array in Node grows linearly** (one object per frame), and the in-page `Engine` keeps only current state — so Node memory
grows with episode length (expected ~1–2 KB/frame from the FrameRecord + PNG write). A long-episode Node-memory check is
still advisable.

### 6.6 [P2] Schema diff: headless manifest vs human playtest manifest

**Headless manifest** (types.ts:214-241) — top-level: `episodeId, renderWidth, renderHeight, outputWidth, outputHeight,
simTimestep, seed, config{…}, frames[]`. Frame (types.ts:200-212): `frameIndex, simulatedTime, imagePath, controlSignalX,
playerX, playerZ, playerHeading, currentEntityId` (**8 fields**, 0-based `frameIndex`).

**Playtest manifest** — top-level: `kind, recordedAt, seed, outputWidth, outputHeight, renderWidth, renderHeight,
simTimestep, config, frames[]` (no `episodeId`). Frame adds, on top of the headless fields:
`playerOmega, lastClickedX, targetDoorwayId, targetPointX, targetPointZ, targetPhase, collisionEvents, wallOverlap,
candidates[], clickedX` (**18 fields**, 1-based `frameIndex`).

**Fields present only in playtest:** `playerOmega, lastClickedX, targetDoorwayId, targetPointX, targetPointZ, targetPhase,
collisionEvents, wallOverlap, candidates, clickedX`, plus `recordedAt`/`kind` at top level.

---

## SECTION 7 — WORLD STATISTICS (RUN, 200 seeds × 1,200 frames × 2 bots)

Method: renderer-free replica of `Engine.step` (the same logic as `tests/regression.test.ts`), `World(seed, CONFIG,
{buildGeometry:false})`. Bots: the real `makePlaceholderBot(seed)` and a scratch **randomReach** bot (picks a uniformly
random reachable doorway each time it enters a new entity). 200 seeds, 1,200 frames each → 240,000 frames per bot.

**Entity taxonomy note:** `junction` and `bend` are **hallway terminations**, not separate entity kinds — an entity is
`kind: 'hallway' | 'room'` and a hallway's far end is a `bend`, `junction`, or `room` doorway (`types.ts:43,117`). So I
report hallways split by termination, which is strictly more informative than the question's "hallway/room/junction/bend".

### 7.1 [P1] Frame share per entity type, and visit lengths

**Placeholder bot:**

| entity type | frames | share |
|---|---|---|
| hallway (bend-terminated) | 51,665 | 21.5% |
| hallway (junction-terminated) | 59,335 | 24.7% |
| hallway (room-terminated) | 42,680 | 17.8% |
| room | 86,320 | 36.0% |
| **hallway total / room** | **64.0% / 36.0%** | |

Visit length (frames) median / p10 / p90: hallway-bend 63/32/131 (n=711); hallway-junction 75/33/140 (n=717);
hallway-room-end 60/31/125 (n=615); room 75/51/99 (n=1023).

**randomReach bot:**

| entity type | frames | share |
|---|---|---|
| hallway (bend) | 55,254 | 23.0% |
| hallway (junction) | 59,064 | 24.6% |
| hallway (room-end) | 42,080 | 17.5% |
| room | 83,602 | 34.8% |
| **hallway total / room** | **65.2% / 34.8%** | |

Visit length median/p10/p90: hallway-bend 63/31/131; hallway-junction 75/32/142; hallway-room-end 62/33/123; room 75/46/98.

### 7.2 [P1] Histograms of visible / reachable candidates per frame

**Placeholder bot** — visible count (0..>6): 0:15,253 · 1:102,280 · 2:64,626 · 3:39,333 · 4:13,405 · 5:4,357 · 6:694 · >6:52.
Reachable count: 0:25,720 · 1:105,946 · 2:62,514 · 3:31,979 · 4:10,153 · 5:3,261 · 6:406 · >6:21.
**Frames with ≥2 reachable candidates: 108,334 (45.1%).**

**randomReach bot** — visible: 0:13,676 · 1:104,202 · 2:65,771 · 3:38,823 · 4:13,021 · 5:4,058 · 6:366 · >6:83.
Reachable: 0:23,462 · 1:108,103 · 2:63,615 · 3:31,851 · 4:9,808 · 5:2,909 · 6:205 · >6:47.
**Frames with ≥2 reachable candidates: 108,435 (45.2%).**

### 7.3 [P1] How often the chosen door is the straightest (|screenX| smallest) when ≥2 reachable

- Placeholder bot: **89.9%** (97,390 / 108,334).
- randomReach bot: **92.8%** (100,606 / 108,435).

(The default no-click rule is "nearest reachable to screen center", which ≈ straightest, so this stays high even under
random clicks because most held clicks also resolve near-center; the randomReach bot is higher because it only ever
clicks reachable doors.)

### 7.4 [P1] controlSignalX histogram (placeholder) + fallback rate

Histogram (0.1-wide bins): **0.0: 121,592 · +1.0: 66,904 · −1.0: 51,504** (only these three bins are non-empty — the
placeholder bot's clicks are uniform in `(-1,1)` and the signal is a **held** step function, so the distribution is a
spike at the default 0 plus the two clamp edges; the interior bins are populated only momentarily between clicks and
average out to near-zero per-bin over 240k frames — see caveat).

**Fallback rate** (click resolved to a *different* doorway than the nearest-by-screen-X): 131 / 4,719 accepted clicks = **2.8%**.

**Caveat on the histogram:** I binned at 0.1; because `controlSignalX` is *held* across frames between sparse clicks
(a click every 20–80 frames), the histogram is dominated by (a) the long no-click default `0` and (b) the held values
near the extremes. A finer histogram of *accepted-click values only* would show a roughly uniform spread in `(-1,1)`.

### 7.5 [P1] Collision / overlap / pinning

| bot | collision events | rate /1k | overlap frames | pinned frames | worst collision/ep | worst overlap/ep |
|---|---|---|---|---|---|---|
| placeholder | 12,348 | 51.45 | 0 | 0 | 996 | 0 |
| randomReach | 10,922 | 45.51 | 0 | 0 | 1,137 | 0 |

"No collision events" here means "frames where the raw next position was off-floor and the slide backstop engaged"
(a wall graze/slide, not a stall). **No episode got pinned or stalled, and there were zero wall-overlap frames.**

### 7.6 [P1] Per-frame heading change (radians/frame)

| bot | mean | p50 | p95 | max | fraction \|ΔH\|<1e-6 |
|---|---|---|---|---|---|
| placeholder | 0.01110 | 0.00081 | 0.06033 | 0.12500 | 18.1% |
| randomReach | 0.01059 | 0.00043 | 0.06302 | 0.12500 | 20.3% |

(Max 0.125 rad/frame = `maxTurnRate` 2.5 rad/s × dt 0.05 s, the turn-rate cap — the jerk limit is not the binding
constraint at steady turn.)

### 7.7 [P2] Steering-phase durations

- Placeholder: approach 207,758 frames (86.6%), through 32,242 (13.4%).
- randomReach: approach 207,522 (86.5%), through 32,478 (13.5%).

(Only two phases exist in code: `approach` and `through` — `navigation.ts:28`. The "turn gate / through-leg" split in
the question is subsumed by these two.)

### 7.8 [P2] Are first frames identical across episodes?

**Direct answer:** No. The start hallway is randomized per seed (`generateStartHallway` picks a random cardinal +
random length + random termination — `layout.ts:650-658`), so first frames vary by seed. Within a fixed seed the first
frame is identical (deterministic). The only per-frame "randomization" at episode start is the seeded world gen; no
time-varying element (flicker off).

---

## SECTION 8 — VISUAL QUALITY AND DATASET HYGIENE

**SKIPPED BY USER.** The user explicitly requested Section 8 be skipped and marked as such. No contact sheets, visual
pop/diversity/artifact checks, or the open-item "visual verification of single-sided walls + wall thickness" were
performed. (That CHANGES-LOG open item therefore remains **unverified** — see 11.3.)

---

## SECTION 9 — HUMAN RECORDING TOOL AND EXISTING DATA

### 9.1 [P1] Everything under /data

The gitignored data dir is **`engine/data/`** (there is no top-level `/data`). Contents:

- `engine/data/playtest_2026-10-09T02-53-19-924Z/` — **one human playtest** (headful recorder), **578 frames**, **48 MB**,
  seed `3969209238`, with a rich per-frame manifest (18 fields, see 9.2). Complete (full `frames/` + `manifest.json`).
- `engine/data/scratch/` — my scratch scripts (to be deleted; see cleanup).

No headless episodes are persisted under `data/` (my generated episodes went to `/tmp` scratch and are deleted).

### 9.2 [P1] Recorder per-frame fields; can the full click list be reconstructed?

**Per-frame fields written** (`main-record.ts:44-65,81-101`): `frameIndex, simulatedTime, imagePath, controlSignalX,
playerX, playerZ, playerHeading, playerOmega, currentEntityId, lastClickedX, targetDoorwayId, targetPointX, targetPointZ,
targetPhase, collisionEvents, wallOverlap, candidates[], clickedX` (plus `dataUrl` drained to PNG).

**Reconstruction: YES.** `clickedX` is the **raw** click applied each frame (including debounced ones), so the full
click-event list (frame index + raw X) **is** reconstructable from a playtest manifest. (The **headless** manifest
cannot — it only records the *accepted* `controlSignalX`.)

### 9.3 [P2] Can per-frame candidate positions + reachability be recovered?

**Yes.** Each playtest frame records `candidates[]` = `[{doorwayId, screenX, reachable, screenDistance}]`
(`main-record.ts:98`), so per-frame candidate screen positions and reachability are available for fitting human-like
bot parameters (reaction delay, click offset, clicks-per-door, mind-change rate).

### 9.4 [P2] Discard-on-close risk to current /data

The open item is real: `record.ts` writes the manifest even when the window is closed without pressing `Q` (it drains
whatever was last polled — `main-record.ts:127`). It does **not** corrupt existing complete data; it only risks leaving
a partial `playtest_<ts>/` behind. The one existing playtest is complete (578 frames + manifest), so nothing currently
in `data/` is contaminated.

---

## SECTION 10 — MACHINE AND ENVIRONMENT (read-only)

### 10.1 [P1] Hardware/OS

- **OS:** Ubuntu 26.04.1 LTS; kernel `7.0.0-38-generic` x86_64.
- **CPU:** Intel(R) Core(TM) Ultra 9 275HX — **24 cores** (nproc).
- **RAM:** **30 GiB**.
- **GPU:** **NVIDIA GeForce RTX 5070 Laptop GPU**, **8,151 MiB** VRAM, driver **595.91.07** (Vulkan 1.4.329).
- **Disk** holding repo+data (`/` on `/dev/nvme1n1p1`): 938 G total, 322 G used, **569 G free (37%)**. `/tmp` is a 16 G tmpfs.

### 10.2 [P1] Python / PyTorch

- **Python:** 3.14.4 (installed).
- **PyTorch:** **NOT installed** (`ModuleNotFoundError: No module named 'torch'`; not in `pip list`). `torch.cuda.is_available()` is N/A.
- **No conda, no virtualenv** in the repo (`.venv`/`venv` absent).

### 10.3 [P2] Node / Chrome

- **Node** v22.22.1; **npm** 12.2.0.
- **Chrome** (Puppeteer): Google Chrome **154.0.8037.97** (`/usr/bin/google-chrome`).

### 10.4 [P2] Is the GPU also driving the display?

This is an Optimus laptop (Intel iGPU + NVIDIA dGPU). Headless generation routes to the NVIDIA dGPU via PRIME offload
(`__NV_PRIME_RENDER_OFFLOAD=1`, `driver.ts:83-86`), while the headful recorder deliberately uses the Intel iGPU
(`--use-angle=gl`, no offload — `driver.ts:98-104`). Heavy headless generation will load the dGPU (measured 654 MiB at
concurrency 8, well under 8 GB) and should not make the desktop unusable, since the display runs on the iGPU.

---

## SECTION 11 — REPO STATE AND OPEN ITEMS

### 11.1 [P1] git status / branch / commit / tags / manifest provenance

- Branch `main`, HEAD **`105f8e4`** ("engine stability improvements"). Working tree **clean** except one pre-existing
  untracked file `planning-questions-for-project-llm.txt` (present before this investigation).
- **No tags** for the engine (`git tag` empty). Manifests do **not** record the git commit or a config hash — they record
  `seed` and a subset of `config` (types.ts:214-241); there is no provenance of code version.

### 11.2 [P1] Test suite + typecheck (RUN)

- `node node_modules/vitest/vitest.mjs run` → **6 files, 52/52 tests passed** (3.93 s).
- `node node_modules/typescript/bin/tsc --noEmit` → **clean (exit 0)**.

### 11.3 [P2] Open items (CHANGES-LOG.md) status

- **Visual verification of single-sided walls + wall thickness:** **still not done** (CHANGES-LOG confirms; §8 skipped by user → remains unverified).
- **Discard-on-close for recordings:** **still open** (record.ts writes manifest even on window close without Q).
- **Corner-hugging steering:** **still open** (aesthetic, tunable via `TURN_RATE_GAIN`/`MAX_TURN_RATE`/`JERK_THRESHOLD`).

**More urgent than those (my judgment):** (1) the `controlSignalX[i]` label off-by-one (§2.2); (2) the missing
"tight-turn gate" that the docs claim exists (§11.4 / §12) — side doorways and junction turns are **not** marked as
turns in code, so the "corner-hugging"/penetration fix described in CHANGES-LOG may not actually be implemented; (3)
path-dependent world generation (§5.1).

### 11.4 [P2] README that is out of date vs code

1. **`README.md:250`** — "`desiredOmega` uses a bang-bang profile `sign(err)·sqrt(2·jerk·|err|)`" — **FALSE**. Code is a
   P-controller `omega = clamp(turnRateGain * headingError, ±maxTurnRate)` then jerk-limited (`steering.ts:159-171`).
2. **`README.md:201` config table** — `DOORWAY_TURN_GATE_RADIUS | 1.5` — **does not exist** in `config.ts`.
3. **`README.md:162`** — example `"currentEntityId": "h0"` — **actual first entity is `h1`** (verified in both a headless
   manifest and the playtest).
4. **`README.md:310-317` "Tight turns use a smaller gate"** and **`CHANGES-LOG.md:130-135`** (per-doorway gate radii,
   `DOORWAY_TURN_GATE_RADIUS=1.5`, side-doorway/junction approach points) — **not present in code**: only **bend** doorways
   set `turn:true` + `approachPoint` (`layout.ts:209-210`); `placeSideDoorways` (`doorway.ts:185-196`) and junction branches
   (`layout.ts:213-228`) create plain doorways, and `gateRadius` is never assigned anywhere (grep confirms only `makeDoorway`
   reads it). Side doorways/junction turns therefore use the same `doorwayGateRadius=2.5` and opening-center approach as
   normal doorways.

---

## SECTION 12 — OPEN-ENDED

### 12.1 [P1] Top five worries before generating 500k training frames

1. **Label/frame misalignment (§2.2).** `controlSignalX[i]` is the value *before* step i; the change i→i+1 is caused by
   the click that only appears as `controlSignalX[i+1]`. A next-frame model conditioned on `controlSignalX[i]` to predict
   frame i+1 uses the **wrong (stale)** signal. Fix by shifting the label, or by logging the click that *will* be applied
   next as the frame's action.
2. **The signal under-specifies steering.** It is a held step-function whose mapping to "which doorway" is re-derived from
   screen projection each time (and re-resolves on entity entry). The same `controlSignalX` can mean different targets at
   different headings/times, and "never clicked" and "clicked center" are both `0` but behave differently (idle drift vs
   sticky center target). This is an ambiguous action space.
3. **World is not a function of seed alone (§5.1).** Lazy generation with a single shared PRNG means entity contents
   depend on the path. Different bots/policies on the same seed generate *different* worlds after their paths diverge,
   and re-entered entities regenerate with different contents. This complicates any cross-policy comparison or "same world,
   different actions" data collection.
4. **Debounced clicks are invisible in the headless manifest (§1.3).** The network cannot learn that a click was ignored;
   rapid multi-clicks collapse to a single accepted value with no trace. If "click density" is meant to be a signal, the
   raw click list must be logged.
5. **Steering robustness of side/junction turns (§11.4).** The docs describe a tight-turn gate (`DOORWAY_TURN_GATE_RADIUS=1.5`)
   that the code does not implement. The wall-contact rate (45–51/1k frames, worst 1,137/episode) is high for a "clean"
   dataset, and corner-hugging is an open item — worth fixing/verifying before bulk generation so the motion is as clean
   as intended.

### 12.2 [P2] What makes click→motion hard to learn

- **Hidden state not in the label:** `angularVelocity`, the current sticky target doorway id, and the `approach`/`through`
  phase all affect the next frame but are not part of the per-frame action signal.
- **Long/variable latency:** median 2 frames to any heading change but **median 10 frames to align** (§2.3), with ~14% of
  clicks never reaching >1° — so the response to a click is spread over many frames and sometimes absent.
- **Ambiguous mapping:** identical `controlSignalX` at different headings → different doorway (doorway screen positions move
  as the player turns), and the held signal persists across entity boundaries re-resolving to a new target (§1.5).
- **Inconsistent responses:** the same click value can be accepted or debounced depending on timing (§3.3), and the target
  may fall back to a non-nearest doorway when the nearest is unreachable.

### 12.3 [P2] Things not asked that the next-stage author should know

- **The bot runs in Node across 3 bridge round-trips/frame** (§4.3); a "smart" bot that needs per-frame candidate data will
  add a `getState()` round-trip unless moved in-page (trivial: `window.__backrooms` already exposes everything).
- **The placeholder bot clicks uniformly random X** (`bot.ts:30`) — it spends ~half its time steering toward walls. For a
  pilot dataset that is fine (it exercises the whole pipeline), but the "teacher" data will be heavily wall-slide/noisy
  until a real policy exists (see the ~45–51 collision events/1k).
- **Second-degree doorways are deterministic solid-black quads** (`World.blackDoorways` + `blackout.ts`), not fog — so the
  "black fill" is exact and predictable, not a rendering artifact.
- **Episode length 600 = 30 simulated seconds** at 20 Hz; entities are visited for ~1.5–3.5 s each (§7.1), so a 600-frame
  episode traverses only ~8–12 entities — a small world slice per episode.

---

## CLEANUP / REPO STATE

- Repo unchanged apart from this file (`engine/docs/PLANNING-ANSWERS.md`, uncommitted). `git status` shows only the
  pre-existing untracked `planning-questions-for-project-llm.txt`.
- Scratch created and deleted:
  - `/tmp/backrooms-scratch/` (bench.sh + `bench/` outputs, determinism episodes `detA`/`detB` + logs) — deleted.
  - `engine/data/scratch/` (`stats.mts`, `timing.mts`, `replay.mts`) — deleted.
