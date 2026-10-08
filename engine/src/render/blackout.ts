/**
 * Second-degree doorway black-fill geometry (spec Section 5.5).
 *
 * A doorway that exists *within* a directly-attached neighbor entity (i.e. a
 * doorway two steps from the player) must read as flat, static, opaque
 * blackness — never a view into anything. This module fills those openings
 * with unlit, fog-free black quads that exactly match each opening.
 */

import * as THREE from 'three';
import { CARDINAL_VEC } from '../core/types';
import type { Doorway } from '../core/types';

/** Slight offset along the doorway normal so the quad never z-fights the
 * surrounding wall plane. */
const INSET = 0.04;

export class BlackoutManager {
  readonly group = new THREE.Group();

  /** One shared unlit black material. `fog: false` guarantees it stays pure
   * black even deep inside the fog; MeshBasicMaterial never responds to light. */
  private readonly material = new THREE.MeshBasicMaterial({
    color: 0x000000,
    side: THREE.DoubleSide,
    fog: false,
  });

  /** Replace all black-fill quads with quads for the given doorways. */
  setDoorways(doorways: Doorway[]): void {
    this.clearQuads();
    for (const doorway of doorways) {
      this.group.add(buildBlackoutQuad(doorway, this.material));
    }
  }

  dispose(): void {
    this.clearQuads();
    this.material.dispose();
  }

  private clearQuads(): void {
    for (const child of [...this.group.children]) {
      this.group.remove(child);
      (child as THREE.Mesh).geometry.dispose();
    }
  }
}

function buildBlackoutQuad(doorway: Doorway, material: THREE.MeshBasicMaterial): THREE.Mesh {
  // Doorway.x/z is the floor-level center of the opening; height is vertical.
  const geometry = new THREE.PlaneGeometry(doorway.width, doorway.height);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;

  const facing = CARDINAL_VEC[doorway.facing];
  // Wall direction = facing rotated +90 degrees in the floor plane; the
  // opening's width runs along the wall.
  const wallDir = { x: -facing.z, z: facing.x };
  const xAxis = new THREE.Vector3(wallDir.x, 0, wallDir.z);
  const yAxis = new THREE.Vector3(0, 1, 0);
  // Front face points back toward the player (-facing), so the fill is visible
  // from inside the current entity. DoubleSide makes orientation moot anyway.
  const zAxis = new THREE.Vector3(-facing.x, 0, -facing.z);
  mesh.setRotationFromMatrix(new THREE.Matrix4().makeBasis(xAxis, yAxis, zAxis));

  // Vertical center = height/2; horizontal center = (x, z), inset along the
  // facing normal so it sits just off the wall plane.
  mesh.position.set(
    doorway.x + facing.x * INSET,
    doorway.height / 2,
    doorway.z + facing.z * INSET,
  );
  return mesh;
}
