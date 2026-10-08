/**
 * Axis-aligned wall collision (pure function, shared by the Engine and tests).
 *
 * The player is treated as a floor point: a move is legal only if it stays
 * inside some entity's floor region (the union of every live entity's interior
 * exactly matches the walkable geometry, because rooms/hallways tile with no
 * gaps and walls are the thin boundaries between floor and void). A move into a
 * wall or the void slides along whichever axis remains on the floor; if neither
 * does (a corner), the player holds position but keeps the new heading so it
 * keeps turning toward its target.
 */
import type { PlayerState } from '../core/types';

export interface FloorQuery {
  containsFloorPoint(x: number, z: number): boolean;
}

export function resolveWallCollision(
  floor: FloorQuery,
  prev: PlayerState,
  next: PlayerState,
  onCollision?: () => void,
): PlayerState {
  if (floor.containsFloorPoint(next.x, next.z)) return next;

  const dx = next.x - prev.x;
  const dz = next.z - prev.z;
  onCollision?.();

  if (Math.abs(dx) > 1e-9 && floor.containsFloorPoint(prev.x + dx, prev.z)) {
    return { ...next, x: prev.x + dx, z: prev.z };
  }
  if (Math.abs(dz) > 1e-9 && floor.containsFloorPoint(prev.x, prev.z + dz)) {
    return { ...next, x: prev.x, z: prev.z + dz };
  }
  return { ...next, x: prev.x, z: prev.z };
}
