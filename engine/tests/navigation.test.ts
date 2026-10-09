/**
 * NAVIGATION / COLLISION REGRESSION TESTS
 * =======================================
 * Plain-Node simulation of the Engine's motion loop (Steering.step +
 * two-phase doorway navigation + resolveWallCollision + hysteresis entity
 * tracking), without a WebGL renderer. These guard the three behaviors a human
 * player depends on most:
 *   1. the player never leaves the floor (never clips through a wall or into
 *      the void),
 *   2. steering toward a hallway side doorway actually carries the player into
 *      the room on the other side, WITHOUT ever clipping a wall, and
 *   3. the entity the player occupies transitions cleanly (once) as they cross.
 */
import { describe, it, expect } from 'vitest';
import { CONFIG } from '../config/config';
import { World } from '../src/world/World';
import { Steering } from '../src/control/steering';
import { resolveWallCollision } from '../src/sim/collision';
import { doorwayApproachPoint, doorwayTargetPoint, doorwayPhase } from '../src/sim/navigation';
import type { DoorwayPhase } from '../src/sim/navigation';
import { CARDINAL_VEC } from '../src/core/types';
import type { Doorway, Entity, PlayerState, Vec2 } from '../src/core/types';

/** Replicates Engine.startingPlayer. */
function startPlayer(start: Entity['descriptor']): PlayerState {
  const dir = start.travelDirection ? CARDINAL_VEC[start.travelDirection] : { x: 1, z: 0 };
  const pt = start.start ?? { x: 0, z: 0 };
  const inset = Math.min(1.5, CONFIG.hallwayLengthMin / 2);
  return {
    x: pt.x + dir.x * inset,
    z: pt.z + dir.z * inset,
    heading: Math.atan2(dir.z, dir.x),
    angularVelocity: 0,
  };
}

/** One simulated step: steer toward `target`, then resolve wall collision. */
function advance(
  world: World,
  steering: Steering,
  player: PlayerState,
  target: Vec2 | null,
  onCollision: () => void,
): PlayerState {
  const stepped = steering.step(player, target, CONFIG.simTimestep);
  return resolveWallCollision(world, player, stepped, onCollision);
}

describe('navigation + collision', () => {
  it('measures a bend doorway reachability against its approach point, not its waypoint', () => {
    // Root-cause regression: a bend's entryPoint is a waypoint TURN_WAYPOINT_DEPTH
    // units PAST the corner. Evaluated from a player approaching the corner, a
    // large waypoint depth sits far to the side and falsely fails the ~75° front
    // cone — the player gets no target and drifts diagonally into the wall. The
    // candidate must instead be reachability-tested against the corner
    // (approachPoint), which is always straight ahead along the corridor.
    const bend: Doorway = {
      id: 'bend',
      x: 20,
      z: 0,
      width: CONFIG.hallwayWidth,
      height: CONFIG.hallwayHeight,
      facing: 'north',
      side: 'right',
      targetEntityId: 'h9',
      targetGenerated: true,
      // Waypoint far past the corner (the magnitude that triggered the crash).
      entryPoint: { x: 20, z: -30 },
      approachPoint: { x: 20, z: 0 },
      turn: true,
    };
    const player = { x: 18, z: 0, heading: 0, angularVelocity: 0 };
    const steering = new Steering(CONFIG);

    expect(doorwayApproachPoint(bend)).toEqual({ x: 20, z: 0 });

    // The corner is straight ahead: reachable.
    expect(steering.isReachable(player, doorwayApproachPoint(bend))).toBe(true);
    // The far waypoint is ~86° off heading: outside the front cone.
    expect(steering.isReachable(player, bend.entryPoint)).toBe(false);
  });

  it('never leaves the floor while wandering and turning into doorways', () => {
    for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {
      const world = new World(seed, CONFIG, { buildGeometry: false });
      const start = world.init();
      world.maintain(start.descriptor.id);
      const steering = new Steering(CONFIG);
      let player = startPlayer(start.descriptor);

      const allDoorways = world
        .all()
        .flatMap((e) => e.descriptor.doorways)
        .filter((d) => d.targetGenerated && d.targetEntityId);

      let targetDoorway: Doorway | null = null;
      let phase: DoorwayPhase = 'approach';
      for (let i = 0; i < 300; i++) {
        // Every 25 frames re-aim at a different doorway (stresses corner turns).
        if (i % 25 === 0 && allDoorways.length > 0) {
          targetDoorway = allDoorways[i % allDoorways.length]!;
          phase = 'approach';
        }
        let target: Vec2 | null = null;
        if (targetDoorway) {
          phase = doorwayPhase(player, targetDoorway, CONFIG, phase);
          target = doorwayTargetPoint(targetDoorway, phase);
        }
        player = advance(world, steering, player, target, () => {});
        expect(
          world.containsFloorPoint(player.x, player.z),
          `seed ${seed} step ${i}: player left the floor`,
        ).toBe(true);
      }
    }
  });

  it('steering into a hallway side doorway reaches its room without clipping', () => {
    for (const seed of [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]) {
      const world = new World(seed, CONFIG, { buildGeometry: false });
      const start = world.init();
      world.maintain(start.descriptor.id);
      const steering = new Steering(CONFIG);

      // Find a side doorway on the start hallway that leads to a generated room.
      const side = start.descriptor.doorways.find(
        (d) =>
          (d.side === 'left' || d.side === 'right') &&
          d.targetGenerated &&
          d.targetEntityId !== '' &&
          world.get(d.targetEntityId)?.descriptor.kind === 'room',
      );
      if (!side) continue; // some seeds have no side doorway; skip

      const roomId = side.targetEntityId;
      let player = startPlayer(start.descriptor);
      let phase: DoorwayPhase = 'approach';
      let collisionEvents = 0;
      let activeId = start.descriptor.id;
      const switches: string[] = [];
      let entered = false;

      for (let i = 0; i < 600; i++) {
        phase = doorwayPhase(player, side, CONFIG, phase);
        const target = doorwayTargetPoint(side, phase);
        player = advance(world, steering, player, target, () => {
          collisionEvents += 1;
        });
        expect(world.containsFloorPoint(player.x, player.z)).toBe(true);

        // Hysteresis: only leave the active entity once it no longer contains
        // the player (mirrors Engine.step).
        if (!world.containsEntity(activeId, player.x, player.z)) {
          const here = world.entityAt(player.x, player.z);
          if (here && here.id !== activeId) {
            activeId = here.id;
            switches.push(activeId);
            if (activeId === roomId) entered = true;
          }
        }
        if (entered) break;
      }

      expect(
        entered,
        `seed ${seed}: never reached the room through side doorway ${side.id}`,
      ).toBe(true);
      expect(
        collisionEvents,
        `seed ${seed}: clipped a wall while entering side doorway ${side.id}`,
      ).toBe(0);
      expect(
        switches,
        `seed ${seed}: entity transition was not a single clean hand-off to the room`,
      ).toEqual([roomId]);
    }
  });
});
