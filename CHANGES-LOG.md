# Changes Log

A running log of changes requested and what happened. **Newest first.** Each
entry records *what was asked → root cause → fix → verification*. For permanent
architecture and subsystem contracts, see [`README.md`](README.md) (definitive)
and `engine/docs/CONTRACTS.md`. This file replaces the old `handoff.md`.

---

## 2026-10-10 — Stage 2: action labels, click-bot personas, resumable runs, pilot

**Asked.** Implement Stage 2 of the training-data generator (brief
`stage2-prompt-bots-labels-pilot.txt`): unambiguous action-aligned labels
(4.1–4.4), a click-bot persona framework whose choices spread uniformly at
*choice moments* rather than always picking the straightest door (4.6), run
plans with resumable split-aware batch runs (4.7), a diagnostics report (4.8),
doc drift fixes (4.9), the pilot dataset + review images (4.10), and the
clipping failsafe (4.11). Simulation behavior (world gen, steering, reachability,
debounce, collision, timing, RNG) was **not** changed.

**Built.** `src/headless/bot.ts` (seven personas + three mixture profiles, bot
RNG separate from world RNG), `src/headless/driver.ts` (v2 manifest, action
alignment, clipping failsafe, scripted `botOverride`), `src/headless/plan.ts`
+ `src/headless/workerPool.ts` (`runPlan`: resume, disjoint seed ranges,
deterministic episode dirs, retry-once, early-stop), `src/headless/cli.ts`
(plan CLI + legacy flags), `src/headless/provenance.ts` (configHash /
engineCommit / dirty / engineSourceHash), `src/headless/report.ts` (the ten
4.8 items), `docs/DATASET-SCHEMA.md` (the data contract). The click pulse
overlay was removed from captured frames (4.5a).

**Key findings.**
- *Section 3.* The earlier "three signal bins (0 / +1 / −1)" observation does
  **not** reproduce: the real placeholder bot's held-signal histogram is spread
  across all 0.1 bins, exactly as the held-signal model predicts. The signal
  path is correct; the old number was an earlier-investigation artifact. No
  engine change.
- *4.6 straightest-target.* The literal per-frame criterion reads **88.7%**
  (above the ~65% threshold) — but this is the per-frame-vs-moments confound
  §3(b) already flags: the `uniform` persona (uniform random X) also reads
  90.2%. Measured at real choice moments (accepted clicks with 2+ reachable),
  the straightest-choice rate is **48.8%** — i.e. choices are spread ~uniformly
  as intended. No tuning applied (see STAGE2-REPORT for the reasoning).

**Verification.** `tsc --noEmit` clean; 56 vitest tests pass (action alignment,
run plans, provenance, failsafe trim + delete paths). Pilot `engine/data/pilot_v1/`:
150 episodes (145 complete + 5 aborted = 3.3%), 88,047 frames, 6.70 GB, 134 s
on the RTX 5070. Diagnostics report and review images written to the pilot dir.

---

## 2026-10-08 — Config-drift world-generation fix + single-sided walls + wall thickness

**Asked (three requests):** (1) make the world generator resilient to the
enlarged room sizes — the 4 failing rule tests must pass without weakening the
tests; (2) walls *and* their blackout doorways should be visible only from the
inside (fixing back-to-back wall z-fighting); (3) give walls a bit of thickness
(visible only in doorways), without ever clipping.

**Root cause (request 1).** Rooms were enlarged to 10–20 while `HALLWAY_LENGTH`
is 8–30 and doorway spacing was `doorwaySpacingFactor * doorwayWidth` = 8.1. A
side-doorway room's "along" dimension (width for E/W travel, depth for N/S)
must fit inside its corridor *cell*, but cells could now be smaller than
`ROOM_WIDTH_MIN`/`ROOM_DEPTH_MIN` (10). `generateRoom` then (a) shrank rooms
below the configured minimum, and (b) produced a degenerate offset region so the
room poked out of its cell into an adjacent room or a crossing corridor. The
seed-13 case was a 9.83-long hallway whose two side rooms (min 10) overflowed
into the parent room at its start and the bend child at its end.

**Fix (request 1).** `placeSideDoorways` (`engine/src/world/doorway.ts`) now
derives `minAlong` from the travel axis and enforces every side room's cell ≥
`minAlong`: consecutive doorways are spaced `max(minSpacing, minAlong)` apart,
each side's cell accounts for the bend turn side (`W/2` at the end) and a bend
child's parent side (`W/2` at the start), and a side whose cell is below
`minAlong` gets **no** doorways (a too-short hallway now yields no side rooms
instead of overlapping ones). `generateHallway` picks the bend turn side up
front and threads `turnSide`/`startCorridorSide` down so placement knows the
back-offs; `generateNeighbor` passes `startCorridorSide` in via `HallwayOpts`.
`generateRoom`'s shrink branch is defensively clamped to never go below the
configured minimum.

**Request 2 (single-sided).** `materials.ts` floor/wall/ceiling changed
`DoubleSide → FrontSide`; `geometry.ts` `addQuad` normalizes winding against the
passed normal so every wall/floor/ceiling front-faces its interior (previously
half the walls faced outward and `DoubleSide` hid it); `blackout.ts` dropped
`DoubleSide` so the fill is culled from the far side. Back-to-back coplanar
walls now each face their own interior and no longer z-fight.

**Request 3 (thickness).** Added `WALL_THICKNESS = 0.5` (`config.ts`);
`geometry.ts` `buildWall` now extrudes jamb/lintel faces **outward** at each
opening so doorways read as having a reveal. Extrusion is outward-only and
collision stays floor-based, so walkable space is unchanged and it can never
clip.

**Decisions (user-confirmed):** `ROOM_HEIGHT_MIN` raised 2.5 → **2.8** (equal to
`DOORWAY_HEIGHT`, so a doorway always fits its wall); wall thickness **0.5**.

**Verification.** `vitest run` 52/52 pass; `tsc --noEmit` clean; a 400-seed ×
80-step sweep (46,882 room instances) found 0 under-min rooms, 0 room–room
overlaps, and 0 room–hallway overlaps. Rendering changes (2 & 3) are
typecheck-clean but **not yet visually verified** — still need a human
playthrough / recorded run.

---

## 2026-10-08 — Wall-crash in a bend hallway (recorded playthrough, seed `3447179621`)

**Asked:** "I'm still able to make the player crash into a wall… around frame 80
[actual: frame 283]… Does it have to do with my anti-sudden-movements,
jerk-restriction code? … Find the root cause whatever it is, I might be wrong."

**Root cause (NOT the jerk restriction).** The hallway `h7` terminates in a
~90° north bend. Its doorway's `entryPoint` is a waypoint
`TURN_WAYPOINT_DEPTH` units **past** the corner — and `TURN_WAYPOINT_DEPTH` had
been hand-tuned from `3` to `30` (the `//3` comment was left as a hint). The
candidate collection in `Engine.collectCandidates` projected and
reachability-tested that doorway against `entryPoint`. From the player
approaching the corner, a waypoint 30 units past the corner sits ~75–86° off
heading, landing **outside** the 75° front cone — so the bend was marked
unreachable. With no reachable candidate, the sticky target resolved to `null`,
and `Steering.step(null)` froze the player's diagonal heading (they had entered
`h7` at ~0.393 rad), drifting them straight into the south wall.

**Fix (two parts, both required for robustness):**
1. Reverted `TURN_WAYPOINT_DEPTH` `30 → 3` (`engine/config/config.ts`).
2. `Engine.collectCandidates` now evaluates candidates against the doorway's
   **approach point** (new helper `doorwayApproachPoint` in
   `engine/src/sim/navigation.ts`) — the corner for a bend, the opening center
   otherwise — for *both* screen-X projection and reachability. This is the
   point the player actually steers toward on the approach leg, so it can never
   falsely fail the front cone regardless of how `TURN_WAYPOINT_DEPTH` is later
   tuned.

**Verification:** new regression test `tests/regression.test.ts` replays the
recorded click sequence through a faithful renderer-free replica of
`Engine.step` and asserts the player navigates the bend with **0** collision
events and never leaves the floor. Plus a unit test in `tests/navigation.test.ts`
("measures a bend doorway reachability against its approach point, not its
waypoint"). `tsc --noEmit` clean.

---

## 2026-10-05 — Four playability bugs (carried over from the old `handoff.md`)

The following four fixes predate this log's introduction and are preserved here
for continuity. Each is guarded by a regression test in
`engine/tests/world.rules.test.ts` or `engine/tests/navigation.test.ts`.

### Dead-end rooms
Rooms could be generated with no reachable exit from their entry (extra doorways
landed on the entry wall or near the entry corner, behind the ~75° cone). Fix:
`placeRoomDoorways` always places the first extra doorway on the wall **opposite**
the entry. Regression test: "every room has a reachable exit doorway from its
entry."

### Extra wall / room overlap
A room on a bend corner poked into the continuation corridor (and its mirror at
the bend child's start), rendering an "extra wall" that vanished on entry. Fix:
`backOffCorner` backs side rooms off by `W/2` from the relevant corridor end;
added `EntityDescriptor.startCorridorSide`. Regression tests: room-vs-room and
room-vs-hallway overlap checks.

### Phantom doorways
A room could get a second doorway on its **entry wall**, occluded by the parent
hallway's solid wall but crossable (floor-based collision). Fix:
`placeRoomDoorways` never places an extra doorway on the entry wall. Regression
test: "never places an extra doorway on a room's entry wall."

### Bend-corner wall-hit + shallow-diagonal doorway penetration
A hallway bend and tight side-doorway/junction turns all sit on a wall the player
drives parallel to, so they must "drive straight, then turn ~90°." Fix: two-phase
turn model — `Doorway.approachPoint` (the corner / corridor centerline abreast of
the opening / junction node) for the approach leg, `entryPoint` (waypoint past
the corner) for the through leg, and per-doorway gate radii
(`TURN_GATE_RADIUS` = 2.5 for bends, `DOORWAY_TURN_GATE_RADIUS` = 1.5 for tight
turns). Regression tests steer the real pipeline through every bend / side
doorway / junction turn and assert perpendicular entry with no pinning.

**Correction (2026-10-10, Stage 2).** The "per-doorway gate radii" description
above is wrong. `DOORWAY_TURN_GATE_RADIUS` does not exist in `config.ts`, and
`Doorway.gateRadius` is never assigned. Only **bend** doorways set `turn: true`
and an `approachPoint`; side doorways and junction turns use the ordinary entry
waypoint. The two-phase model therefore applies to bends, and tight side /
junction turns are not gated. The README config table was corrected accordingly.

---

## Open items / known issues

- **Visual verification of single-sided walls + wall thickness (not yet done).**
  Backface culling and the 0.5-unit outward wall extrusion are implemented and
  typecheck-clean, but need a human playthrough / recorded run to confirm: walls
  render from inside only, no z-fighting at doorways, doorways show a visible
  jamb, and walking through a doorway shows no clipping or see-through.

- **Discard-on-close for recordings.** `engine/src/headless/record.ts` writes
  the manifest even when the player closes the window without pressing Q. Fix:
  write to a temp dir and only keep/rename on Q.

- **Corner-hugging steering (aesthetic polish).** Turns are functionally correct
  (perpendicular entry, no clipping); the remaining ask is cosmetic "don't hug
  the corner as close as possible," tunable via `TURN_RATE_GAIN` /
  `MAX_TURN_RATE` / `JERK_THRESHOLD` in `engine/src/control/steering.ts`.

---

## Operational notes (recurring gotchas)

- `npx` is broken in this environment — invoke binaries directly (see README
  "Testing"): tests `node node_modules/vitest/vitest.mjs run`, typecheck
  `node node_modules/typescript/bin/tsc --noEmit`, build
  `node node_modules/vite/bin/vite.js build`, scripts
  `node node_modules/tsx/dist/cli.mjs <file>`. Run these from `engine/`.
- The `Read` tool cannot view the PNG frame captures; inspect a frame by
  delegating to a vision-capable subagent, or ask the user (they describe frames
  precisely by frame number).
- `World` is lazy: to replay a recorded path you must *walk* it —
  `world.init()`, then `world.maintain(active.id)` and follow the doorway whose
  `targetEntityId` is the next id. Never `maintain` a non-active id directly.
- Recording manifests record `seed` + per-frame click sequence but not room
  coordinates; regenerate `new World(seed, CONFIG, { buildGeometry: false })`
  to reconstruct geometry.
- Headful recorder (`npm run record`) uses different Chrome launch args than the
  headless pipeline (`HEADFUL_LAUNCH_ARGS` vs `GPU_LAUNCH_ARGS` in
  `engine/src/headless/driver.ts`) — Vulkan/ANGLE has no surface in a windowed
  X11 session.
