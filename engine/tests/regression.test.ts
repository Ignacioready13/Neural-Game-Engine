/**
 * REGRESSION TEST — the wall-crash bug (recorded playthrough, seed 3447179621)
 * ==========================================================================
 * The player crashed into a corridor wall while navigating a hallway that
 * terminates in a ~90° bend. Root cause: candidates were projected from and
 * reachability-tested against the bend's `entryPoint` — a waypoint several
 * units PAST the corner (`TURN_WAYPOINT_DEPTH`). From a player approaching the
 * corner, that waypoint sits far to the side and lands outside the ~75° forward
 * cone, so the bend was marked unreachable, no target resolved, and the frozen
 * diagonal heading drifted the player into the wall.
 *
 * The fix evaluates candidates against the doorway's approach point (the
 * corner). This test replays the recorded click sequence through a faithful,
 * renderer-free replica of Engine.step and asserts the player navigates the
 * bend with zero wall collisions and never leaves the floor.
 */
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { CONFIG } from '../config/config';
import { World } from '../src/world/World';
import { Steering } from '../src/control/steering';
import { resolveWallCollision } from '../src/sim/collision';
import { doorwayApproachPoint, doorwayPhase, doorwayTargetPoint } from '../src/sim/navigation';
import type { DoorwayPhase } from '../src/sim/navigation';
import { verticalFovDeg } from '../src/render/scene';
import { CARDINAL_VEC } from '../src/core/types';
import type { CandidateDoorway, Doorway, PlayerState, Vec2 } from '../src/core/types';

const SEED = 3447179621;
// The recorded clicks (frame -> normalized screen X) that drove the player into
// the bend hallway and triggered the crash.
const CLICKS: Record<number, number> = {
  112: -0.32044522554188637,
  128: 0.8558875219683655,
  157: -0.18765866041788715,
  235: -0.8437805116188244,
};

describe('recorded bend-crash regression', () => {
  it('navigates the bend hallway with zero wall collisions and never leaves the floor', () => {
    const world = new World(SEED, CONFIG, { buildGeometry: false });
    const steering = new Steering(CONFIG);
    const aspect = CONFIG.renderWidth / CONFIG.renderHeight;
    const camera = new THREE.PerspectiveCamera(
      verticalFovDeg(CONFIG.fovHorizontalDeg, aspect),
      aspect,
      0.02,
      200,
    );

    const start = world.init();
    let activeId = start.descriptor.id;
    world.maintain(activeId);
    const dir = start.descriptor.travelDirection
      ? CARDINAL_VEC[start.descriptor.travelDirection]
      : { x: 1, z: 0 };
    const pt = start.descriptor.start ?? { x: 0, z: 0 };
    const inset = Math.min(1.5, CONFIG.hallwayLengthMin / 2);
    let player: PlayerState = {
      x: pt.x + dir.x * inset,
      z: pt.z + dir.z * inset,
      heading: Math.atan2(dir.z, dir.x),
      angularVelocity: 0,
    };

    let targetDoorwayId: string | null = null;
    let phase: DoorwayPhase = 'approach';
    let currentTargetPoint: Vec2 | null = null;
    let collisionEvents = 0;
    const visited = new Set<string>([activeId]);

    const project = (wx: number, wz: number, wy: number): number | null => {
      camera.position.set(player.x, CONFIG.eyeHeight, player.z);
      const fx = Math.cos(player.heading);
      const fz = Math.sin(player.heading);
      camera.lookAt(player.x + fx, CONFIG.eyeHeight, player.z + fz);
      camera.updateMatrixWorld(true);
      const v = new THREE.Vector3(wx, wy, wz);
      v.applyMatrix4(camera.matrixWorldInverse);
      if (v.z >= -1e-9) return null;
      v.applyMatrix4(camera.projectionMatrix);
      return v.x;
    };

    for (let i = 0; i < 400; i++) {
      const dt = CONFIG.simTimestep;
      if (CLICKS[i] !== undefined) steering.handleClick(CLICKS[i]!, i * dt);

      const active = world.get(activeId);
      const candidates: CandidateDoorway[] = [];
      if (active) {
        const reference = steering.lastClickedX ?? 0;
        for (const d of active.descriptor.doorways) {
          // Mirror Engine.collectCandidates: project + reachability-test the
          // APPROACH point (the corner for a bend), not the entryPoint waypoint.
          const approach = doorwayApproachPoint(d);
          const sx = project(approach.x, approach.z, CONFIG.eyeHeight);
          if (sx === null) continue;
          candidates.push({
            doorwayId: d.id,
            screenX: sx,
            reachable: steering.isReachable(player, approach),
            screenDistance: Math.abs(sx - reference),
          });
        }
      }

      const liveIds = active ? active.descriptor.doorways.map((d) => d.id) : [];
      const targetId = steering.update(candidates, liveIds, i * dt);
      if (targetId !== targetDoorwayId) {
        targetDoorwayId = targetId;
        phase = 'approach';
      }

      const targetDoorway: Doorway | null =
        active && targetId !== null
          ? active.descriptor.doorways.find((d) => d.id === targetId) ?? null
          : null;
      if (targetDoorway) {
        phase = doorwayPhase(player, targetDoorway, CONFIG, phase);
        currentTargetPoint = doorwayTargetPoint(targetDoorway, phase);
      } else {
        currentTargetPoint = null;
        phase = 'approach';
      }

      const stepped = steering.step(player, currentTargetPoint, dt);
      player = resolveWallCollision(world, player, stepped, () => {
        collisionEvents += 1;
      });

      expect(
        world.containsFloorPoint(player.x, player.z),
        `frame ${i}: player left the floor (active=${activeId})`,
      ).toBe(true);

      // Hysteresis entity tracking (mirrors Engine.step).
      if (!world.containsEntity(activeId, player.x, player.z)) {
        const here = world.entityAt(player.x, player.z);
        if (here && here.id !== activeId) {
          activeId = here.id;
          visited.add(activeId);
          world.maintain(activeId);
        }
      }
    }

    expect(collisionEvents).toBe(0);
    expect(visited.size).toBeGreaterThanOrEqual(3); // r6 -> h7 (bend) -> h9
  });
});
