/**
 * Doorway navigation target selection (pure functions).
 *
 * Point-and-click steering must route the player THROUGH a doorway opening
 * without ever clipping a wall. Because every entity is a convex axis-aligned
 * box and a doorway is an opening in one of its walls, the safe strategy is
 * two-phase:
 *
 *   approach — steer toward a point just inside the entity, on the doorway's
 *     inward normal. A straight line from anywhere in the (convex) entity to
 *     this point stays on the floor, so the player can never hit a wall while
 *     approaching. This carries the player down the corridor / across the room
 *     until they are abreast of the opening.
 *
 *   through  — once the player is close enough to the approach point, commit to
 *     a point just past the opening (the doorway's entryPoint) and turn through
 *     it. Because the turn starts abreast of the opening center, the turn arc
 *     stays inside the opening.
 *
 * The wall-collision slide (sim/collision.ts) remains purely a backstop; this
 * module's job is to choose good targets so it is never invoked.
 */
import type { EngineConfig } from '../../config/config';
import type { Doorway, PlayerState, Vec2 } from '../core/types';
import { vecDist } from '../core/math';

/** Which leg of the two-phase approach the player is on for a given doorway. */
export type DoorwayPhase = 'approach' | 'through';

/** The world-space steering target for a doorway at a given phase. */
export function doorwayTargetPoint(
  doorway: Doorway,
  phase: DoorwayPhase,
): Vec2 {
  // A turning doorway (a hallway bend) sits on a wall the player drives parallel
  // to, so it needs a ~90 degree turn. It sets an approachPoint (the corner) so
  // the player keeps driving straight down the corridor until abreast of the
  // corner, then commits to 'through' and steers at the waypoint (entryPoint)
  // past the corner, which turns them ~90 degrees.
  if (doorway.turn) {
    if (phase === 'through') return doorway.entryPoint;
    return doorway.approachPoint ?? { x: doorway.x, z: doorway.z };
  }
  if (phase === 'through') return doorway.entryPoint;
  // approach: steer at the opening center (on the wall, at the opening). A
  // pursuit curve from anywhere in the convex entity toward a boundary point at
  // the opening stays inside the entity, so approaching can never clip a wall —
  // and because the opening is laterally offset from the corridor centerline,
  // the player makes a clear, visible turn toward it (unlike an over-pulled-back
  // waypoint near the centerline, which read as "ignoring the click").
  return { x: doorway.x, z: doorway.z };
}

/** Advance the doorway phase for this frame's player position. Once committed
 * to 'through', stay there until the doorway is left behind (the engine resets
 * the phase when the target doorway changes). */
export function doorwayPhase(
  player: PlayerState,
  doorway: Doorway,
  config: EngineConfig,
  phase: DoorwayPhase,
): DoorwayPhase {
  if (phase === 'through') return 'through';
  const approach = doorwayTargetPoint(doorway, 'approach');
  // A perpendicular turn (a bend) needs a larger gate radius than a normal
  // doorway: it is a full ~90 degree turn, so the player must commit several
  // units before the corner to finish the turn by the time they cross into the
  // neighbor.
  const gate =
    doorway.gateRadius ??
    (doorway.turn ? config.turnGateRadius : config.doorwayGateRadius);
  return vecDist(player, approach) <= gate ? 'through' : 'approach';
}
