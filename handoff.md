# Handoff — Phase 1 backrooms engine debugging

Repo root: `/home/ignac/NeuralGameEngine` (code in `engine/`). NOT a git repository.

## Overall state

The engine works well. The user playtested and said: *"MUCH better thanks. It's usable now."* then *"By the way it works pretty dang well."* The movement/steering/world-gen are all functional; the remaining work is polish plus a couple of small plumbing fixes.

Over four follow-up playthroughs the user found and I fixed four real playability bugs — **dead-end rooms**, **phantom doorways**, **bend corners that the player plows straight through into a wall**, and **shallow-diagonal penetration of side doorways/junction turns** (all documented below). Each is covered by a regression test.

## What I fixed this session (verified)

### The dead-end room bug (highest priority, playtest `playtest_2026-10-05T02-18-13-646Z`)

**Symptom:** in seed `416396227`, the player entered room `r25` (via hallway `h25`'s south side-doorway) heading south and then ran straight into the south wall and stopped — the room had no doorway ahead, and the only extra doorway was on the east wall near the entry corner (behind the player, unreachable), so the player had no reachable target and the game dead-ended (~frame 1390 onward; stuck at z≈62.27).

**Root cause:** `placeRoomDoorways` (`engine/src/world/layout.ts`) placed the room's extra doorway(s) on *any* wall, including the entry wall (literally "back the way they came") and perpendicular walls near the entry corner. The player walks into a room perpendicular to the entry wall and can only turn within the ~75° reachability cone (never 180°), so such doorways are behind/off-cone and the player walks into a blank wall. Detected this structurally: **505 of 13,150 generated rooms (~3.8%) had no reachable exit from their entry** across the current code.

**Fix:** `placeRoomDoorways` now always places the first extra doorway on the wall **opposite** the entry (guaranteed ahead/reachable), then optionally adds 0–2 more on any wall. Documented in README judgment calls.

**Verification:** new regression test "every room has a reachable exit doorway from its entry" in `engine/tests/world.rules.test.ts` (uses `Steering.isReachable` from the entry state). Full suite **47/47 passing**, `tsc --noEmit` clean. Re-checked seed `416396227`, `233544193`, and `13` — 0 dead-ends each.

### The "extra wall" / room-overlap bug

**Symptom (user's exact words):** entering one specific room, the player saw an EXTRA WALL (with an extra doorway) inside the room; the instant they stepped through the threshold, the extra wall vanished, revealing the true blank wall further back. Classic "overlapping rooms."

**Root cause (two distinct, symmetric defects, both in `engine/src/world/layout.ts`):**

The world is generated lazily: only the active entity + its direct neighbors are live at any time. A **bend** (90° hallway turn) places its continuation corridor *centered on the parent's end point*, so that corridor reaches `W/2` *back into* the parent's final stretch. A side-room on the turn side that extended all the way to the corridor end therefore poked into the continuation corridor. Its wall rendered "inside" the room, and the moment the player entered (which disposes the no-longer-neighbor corridor), the wall disappeared.

Concretely, for seed `233544193`: hallway `h50` (traveling west) bent south into `h52`. Room `r61` (north side of `h50`, entered via doorway `d384`) had west edge `-13.37`, overlapping `h52`'s corridor band `x ∈ [-16.10, -12.10]`. `h52`'s east wall at `x=-12.10` was the "extra wall."

The mirror-image defect also existed at the **start** of a bend child: room `r4` (seed `13`) on `h2`'s south side poked west back into parent `h1`'s corridor.

**Fix, in `generateNeighbor`'s side-doorway branch (`layout.ts`):** a new helper `backOffCorner(range, cornerCoord, interiorSign, W)` backs the room's `alongRange` off by `W/2` from the relevant corridor end:

- **END side** (parent terminates in `bend`): back off turn-side rooms from the end. Turn side is derived from the parent's `end` doorway `facing` vs `leftOf/rightOf(travelDirection)`.
- **START side** (this hallway itself entered via a bend): back off parent-side rooms from the start. The parent's side is stored on the descriptor as a new optional field `startCorridorSide?: 'left' | 'right'` (set in the bend-child branch: `rightOf(parentDir)` → `'right'`, else `'left'`).

**Second required change:** backing off at the start exposed that the bend child's `startClearance` (`W/2 + dw/2 = 3`) was too small — the gap between the parent corridor (reaches `W/2` back) and the next same-wall doorway could fall below `roomDepthMin` (4), forcing `generateRoom` to shrink a room below its configured minimum (caught a room at depth `3.70`). Changed the bend child's `startClearance` to match the bend end-clearance: `hallwayWidth/4 + minSpacing(config) - doorwayWidth/2` (== 5), symmetric with the end side.

**Files changed this session:**
- `engine/src/world/layout.ts` — the fix + `backOffCorner` helper + `startCorridorSide` assignment + `startClearance` bump.
- `engine/src/core/types.ts` — added `startCorridorSide?: 'left' | 'right'` to `EntityDescriptor`.
- `engine/tests/world.rules.test.ts` — added a regression test "never places a room that overlaps a hallway corridor" (walks 60 seeds × 40 steps, checks every live room against every live hallway's axis-aligned band for positive-area overlap).

**Verification:** `tsc --noEmit` clean; full suite **46/46 passing** (`node node_modules/vitest/vitest.mjs run`). The new room-vs-hallway test plus the existing room-vs-room test both pass. The original seed's overlap (`r61 x h52`) is gone.

### The phantom-doorway bug (second follow-up, playtest `playtest_2026-10-05T02-46-01-946Z`)

**Symptom:** at ~frame 160 the player turned into a room that appeared to have no exit; its only extra doorway was on the **same wall the player entered through**, and that opening was occluded by the parent hallway's solid wall — so it wasn't visible, but the engine's floor-based collision never caught the player crossing it (the floor is contiguous on both sides of the wall). The player "phased" through the phantom opening in frames 214–217 with `collisionEvents=0` throughout.

**Root cause:** `placeRoomDoorways`'s "extra doorways (0–2 on random walls)" step iterated `shuffled(rng, [0,1,2,3])` — *including the entry wall*. The entry wall is shared with the parent hallway, which has exactly one opening (the entry doorway itself), so any second doorway there is a phantom: an opening in the room's wall with solid hallway wall directly behind it. Detected structurally: **595 of 20,798 rooms had an additional doorway on their entry wall**.

**Fix:** `placeRoomDoorways` now computes `otherWallIndices` (all walls except the entry side) and uses it for both the "extra doorways" step and the safety net, so a room can never receive a second doorway on the wall the player entered through. The guaranteed opposite-wall doorway is unchanged.

**Also fixed (the reason this slipped through):** the playtest recorder (`npm run record`) and generator (`npm run generate`) only rebuilt `dist/` when the target HTML was missing or `--build` was passed, so the playthrough was recorded against a **stale vite bundle** built from pre-fix source. Added `engine/src/headless/stale.ts` (`needsRebuild`) and wired both CLIs to auto-rebuild when any source/config file is newer than the built bundle (see "Remaining work" item 4).

**Verification:** new regression test "never places an extra doorway on a room's entry wall" in `engine/tests/world.rules.test.ts`. Full suite **48/48 passing**, `tsc --noEmit` clean. Re-swept 300 seeds (21,222 rooms): **0** rooms with an extra doorway on the entry wall (was 595). The user's seed `3108737186` now yields `extraOnEntryWall=0` for every room reached.

### The bend-corner wall-hit bug (third follow-up)

**Symptom (user's exact words):** *"I ran into a wall again. This time it was a regular hallway that just hooks right, I didn't see anything wrong with it, but when I approached the corner I just kept going straight and hit the wall."* Approached the corner ~frame 1513, hit the wall ~frame 1538 and every frame after.

**Root cause (two compounding defects in bend navigation):** a hallway bend puts its doorway on a **side wall** (perpendicular to the player's approach), unlike a side doorway (approached perpendicular) or a junction branch (which has a W×W node). The old navigation treated it like a normal doorway, and both phases were wrong:

1. **Wrong approach target.** `doorwayTargetPoint` returned the bend's `entryPoint` (a waypoint in the *child* corridor, off to the turn side) for the whole maneuver. Steering toward it pulled the player diagonally toward the turn-side wall — and into any side doorway/room sitting there — before they ever reached the corner.
2. **Gate radius far too small.** The player only committed to the ~90° turn at `DOORWAY_GATE_RADIUS` (1.25 units) from the corner. A right-angle turn needs several units of run-up; at 1.25 the player crossed into the new corridor while still heading down the parent corridor. The bend doorway then left the live set, the target re-resolved to null (the child's exit was behind/unreachable mid-turn), and the player plowed straight into the new corridor's far wall.

**Fix:** bends now use a proper two-phase turn (mirrors the normal doorway model, but for a 90° corner):

- `Doorway` gained `approachPoint?: Vec2`; a bend sets it to the parent's **corner** (the centerline `end`). `navigation.doorwayTargetPoint` steers toward `approachPoint` while approaching (keeps the player driving straight down the corridor) and toward `entryPoint` (the waypoint past the corner) once committed.
- `navigation.doorwayPhase` uses a dedicated `config.turnGateRadius` (**2.5**, up from 1.25) for bends, so the turn starts with enough run-up to finish by the corner.

**Files changed:**
- `engine/src/core/types.ts` — added `Doorway.approachPoint?: Vec2` (and clarified `bendTurn`).
- `engine/src/world/doorway.ts` — `DoorwaySpec`/`makeDoorway` carry `approachPoint`.
- `engine/src/world/layout.ts` — bend branch sets `approachPoint` to the corner and `entryPoint` to the waypoint.
- `engine/src/sim/navigation.ts` — bend-aware `doorwayTargetPoint` + `bendGateRadius` in `doorwayPhase`.
- `engine/config/config.ts` — added `TURN_GATE_RADIUS = 2.5` and `TURN_WAYPOINT_DEPTH = 3.0` (originally landed as `BEND_GATE_RADIUS`/`BEND_WAYPOINT_DEPTH`, later renamed when the same two-phase model was extended to side doorways and junction turns).
- `engine/tests/world.rules.test.ts` — regression test "steers through every bend into the new corridor without hitting the far wall".

**Verification:** the regression test simulates the REAL pipeline (`Steering.step`, `doorwayPhase`, `doorwayTargetPoint`, `resolveWallCollision`) approaching every generated bend and asserts the player turns into the child corridor (heading within 45° of the child's travel direction) and never gets pinned. Full suite **49/49 passing**, `tsc --noEmit` clean. Swept **2000 seeds → 24,376 bends, 0 failures** (previously the waypoint-only fix failed 159/2598 bends).

### The shallow-diagonal doorway-penetration bug (fourth follow-up)

**Symptom (user's exact words):** *"Sometimes the player approaches a doorway almost parallel to the wall it's embedded in, and by the time they get to the doorway they are almost strafing the wall... The camera doesn't turn enough, and the camera goes through the doorway at a super sharp diagonal angle... causing them to make it through the doorway but crash into whatever wall is directly perpendicular to the doorway."* Penetrated a doorway ~frame 438 looking into the door frame's side instead of through it, crashed into the perpendicular wall ~frame 450.

**Root cause:** a **side doorway** (hallway → room) and a **junction branch that turns left/right** both put their doorway on a side wall, so the player drives parallel to that wall and must turn ~90° to enter. The old navigation steered the *approach* phase at the doorway's opening center (the old `doorwayTargetPoint` default), which pulls the player toward the turn-side wall on a pursuit curve. A ~90° turn needs ~3.5 units of travel, but the turn was back-loaded: by the time the player reached the opening they'd only turned ~28–45° off the corridor heading, so they crossed the 2-wide opening at a shallow diagonal and clipped the wall perpendicular to the doorway. (Bends already avoided this via their corner approach point; the fix here extends the same model to the two tight-turn cases.)

**Fix — "drive straight, then turn":**
- A side doorway now sets `approachPoint` to the **corridor centerline abreast of the opening** (not the opening center), so the approach keeps the player driving straight down the corridor instead of drifting toward the turn-side wall.
- A junction turn sets `approachPoint` to the **junction node center** (`end + dir·W/2`), for the same reason.
- Both now set `Doorway.gateRadius = config.doorwayTurnGateRadius` (**1.5**), committing the turn close enough that the turn arc stays inside the 2-wide opening and the player crosses essentially perpendicular. A bend keeps the larger `turnGateRadius` (**2.5**), since it turns through a full-width opening with no wall at the corner.

**Files changed:**
- `engine/config/config.ts` — added `DOORWAY_TURN_GATE_RADIUS = 1.5`; clarified `TURN_GATE_RADIUS` (bends) and `TURN_WAYPOINT_DEPTH`.
- `engine/src/core/types.ts` — added `Doorway.gateRadius?: number` (per-doorway turn-gate override).
- `engine/src/world/doorway.ts` — `DoorwaySpec.gateRadius`; side doorways set `approachPoint`/`gateRadius`/`turn`.
- `engine/src/world/layout.ts` — junction-turn branch sets `approachPoint` (node center) + `gateRadius`.
- `engine/src/sim/navigation.ts` — `doorwayPhase` gate = `doorway.gateRadius ?? (doorway.turn ? turnGateRadius : doorwayGateRadius)`.
- `engine/tests/world.rules.test.ts` — regression test "steers through every side doorway and junction turn perpendicular to the opening".

**Verification:** regression test drives the real pipeline (`Steering.step`, `doorwayPhase`, `doorwayTargetPoint`, `resolveWallCollision`) through every generated side doorway and junction turn, asserting the player enters the target heading within 20° of the doorway's facing and is never pinned. Full suite **50/50 passing**, `tsc --noEmit` clean. Swept `_verify.ts`: side-doorway penetration error p50 **3.1°** (was ~28°), junction-turn error p50 **3.1°** (was ~45°), bend error p50 21.4° (unchanged), all with **0** real crashes.

## Remaining work (not started this session — the user's full ask)

The user's message listed five things. Only the overlap bug (item 2) is done. The other four:

1. **Fix clipping ("I clipped and kept going").** NOT investigated this session. There is collision detection somewhere (the recording manifest records `collisionEvents` and `wallOverlap` per frame, so the logic exists — I did not locate the file). Note: a *previous* session's handoff mentioned a near-plane clipping issue (`near plane = 0.1` in `scene.ts:76`) causing walls to clip when walking ~0.04 units from them — that is a *different* "clipping" (visual near-plane), possibly relevant, possibly unrelated to "clipped and kept going" (which sounds like colliding with a wall and sliding/continuing). Both deserve a look; do not conflate them. **NOTE (this session's second bug, done):** the "phantom doorway" clip below is a *third* distinct kind of "clipping" (passing through an opening that is occluded by the parent hallway's wall, with zero collision because collision is floor-based). It is now fixed — see "The phantom-doorway bug" above.

2. ✅ **Overlapping rooms / extra wall** — DONE (above).

3. **Corner-hugging steering (polish).** The functional bug is fixed (see "The bend-corner wall-hit bug" and "The shallow-diagonal doorway-penetration bug" above): bends, side doorways, and junction turns all now drive straight down the corridor then turn ~perpendicular to the opening. What remains is the *aesthetic* "don't hug the corner as close as possible" polish the user originally asked for. Relevant files: `engine/src/control/steering.ts` (heading/omega/jerk-limited P-controller; `TURN_RATE_GAIN=3.0`, `MAX_TURN_RATE=2.5`, `JERK_THRESHOLD=6.0`), `engine/src/sim/navigation.ts` (`doorwayTargetPoint`/`doorwayPhase`, now bend/side/junction-aware), `engine/src/sim/Engine.ts` (steering.update with `liveDoorwayIds` sticky target).

4. ✅ **Config/source propagation (`config.ts` should apply everywhere, incl. recorder + data generators).** DONE this session. Root cause was: both `engine/src/headless/record.ts` and `engine/src/headless/cli.ts` only rebuilt `dist/` when the target HTML was missing OR `--build` was passed, so editing any source/config file left a **stale vite bundle** — the recorder/generators silently ran old code (this is exactly why the phantom-doorway playthrough was recorded against pre-fix code). Added `engine/src/headless/stale.ts` (`needsRebuild(root, targetHtml)` — true when `dist/<targetHtml>` is missing or older than any file under `config/`/`src/` or the entry HTMLs/vite.config/package.json/tsconfig). Both CLIs now call `needsRebuild` instead of `existsSync`, so they auto-rebuild whenever the source has changed.

5. **Discard-on-close for recordings.** Root cause identified: `engine/src/headless/record.ts` ALWAYS writes the manifest (and already drained PNGs) whether the player pressed Q or just closed the window. The page sets `window.__recordDone = true` only on Q; the poll loop already distinguishes `done` (Q) vs `closed` (window closed) but then unconditionally writes. Fix: only write the manifest when `done` is true; on close-without-Q, delete the already-written `frames/*.png` + the output dir (or write to a temp dir and rename on success).

## Key environment facts (recurring gotchas)

- **The Read tool CANNOT view the PNG frames** — the user confirmed "YOU DO NOT HAVE THE ABILITY TO VIEW IMAGES." To inspect a frame, delegate to a Sonnet subagent (vision) and describe the frame explicitly. The user will also describe frames precisely (they gave exact frame numbers for the overlap bug).
- **Headful recorder launch args** differ from headless: `HEADFUL_LAUNCH_ARGS` (`--use-angle=gl`, no NVIDIA offload env) in `driver.ts` — Vulkan/ANGLE has no surface in a windowed X11 session and yields a blank window. `GPU_LAUNCH_ARGS` (`--use-angle=vulkan` + `GPU_ENV` PRIME offload) is for the headless data pipeline only.
- **`npx` is broken** in this environment. Use direct node entry points:
  - tests: `node node_modules/vitest/vitest.mjs run`
  - typecheck: `node node_modules/typescript/bin/tsc --noEmit`
  - build: `node node_modules/vite/bin/vite.js build`
  - scripts: `node node_modules/tsx/dist/cli.mjs <file>`
- **Run from `engine/`** for the above (the `node_modules` live there), though the repo root is `/home/ignac/NeuralGameEngine`.
- **Recording manifest fields** (per frame): `frameIndex, simulatedTime, imagePath, controlSignalX, playerX, playerZ, playerHeading, playerOmega, currentEntityId, lastClickedX, targetDoorwayId, targetPointX, targetPointZ, targetPhase, collisionEvents, wallOverlap, candidates, clickedX`. It does **NOT** include room coordinates/dimensions — to reconstruct geometry you must regenerate `new World(seed, CONFIG, { buildGeometry: false })` from the manifest's `seed` and replay the entity sequence.
- **`World` is lazy**: `world.maintain(activeId)` generates the active entity's ungenerated neighbors then disposes everything that isn't active or a direct neighbor. To replay a recorded path you must actually *walk* it (start at `world.init()`, then for each entity in sequence: `maintain(active.id)`, find the doorway whose `targetEntityId` is the next id, `active = world.get(next)`). You cannot call `maintain('r61')` directly.
- **Memory rule (must persist):** never push to remote — commit locally, never `git push`.
