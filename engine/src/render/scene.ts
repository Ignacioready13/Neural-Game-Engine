/**
 * Scene graph, camera, lighting, fog, and flicker for the backrooms renderer.
 *
 * The world generator (src/world) owns all surface geometry and materials; this
 * module owns everything that makes those surfaces read as a lit, fogged,
 * "liminal institutional" space: the perspective camera, overhead
 * fluorescent-style point lights on a grid, a gentle deterministic flicker,
 * ambient fill so corners never go pitch black, and depth fog.
 */

import * as THREE from 'three';
import type { EngineConfig } from '../../config/config';
import type { EntityDescriptor, PlayerState } from '../core/types';
import { CEILING_TILE, ceilingAnchors, snapToCeilingTile } from '../world/materials';

/** How far below a ceiling the fixture lights hang (world units). Purely
 * cosmetic — kept out of config because it is not a behavioral knob. */
const LIGHT_DROP = 0.35;

/** Side length of a square ceiling-light panel, as a fraction of a ceiling tile
 * (a recessed troffer sits inside the tile with a thin grout border around it). */
const PANEL_SIZE = CEILING_TILE * 0.84;

/** Emissive brightness of a ceiling-light panel (matches the point light so both
 * flicker together). */
const PANEL_EMISSIVE = 1.0;

/** Point-light falloff exponent. 1.0 = gentle linear falloff that lights a
 * whole room evenly without washing it out (quadratic looked too dark at
 * 8-bit output). */
const LIGHT_DECAY = 1.0;

/**
 * Derive the vertical field of view (degrees) that produces the requested
 * horizontal FOV at a given aspect ratio (width / height). Pure helper.
 * tan(v/2) = tan(h/2) / aspect.
 */
export function verticalFovDeg(horizontalFovDeg: number, aspect: number): number {
  const h = (horizontalFovDeg * Math.PI) / 180;
  const v = 2 * Math.atan(Math.tan(h / 2) / aspect);
  return (v * 180) / Math.PI;
}

/** Deterministic per-fixture phase in [0, 2pi). An irrational multiplier mod
 * 2pi de-correlates fixtures without any Math.random call. */
function fixturePhase(index: number): number {
  return (index * 2.399963229728653) % (Math.PI * 2);
}

/** Smooth, mild, deterministic flicker multiplier for one fixture at time t.
 * Sum of three incommensurate sines -> gentle wander, never a hard strobe.
 * Result is in [1 - flickerIntensity, 1]. */
function flickerMultiplier(config: EngineConfig, phase: number, t: number): number {
  if (!config.flickerEnabled) return 1;
  const w = Math.PI * 2 * config.flickerFrequencyHz;
  const n =
    (Math.sin(w * t + phase) +
      Math.sin(w * 0.383 * t + phase * 1.7) +
      Math.sin(w * 1.618 * t + phase * 0.53)) /
    3;
  return 1 - config.flickerIntensity * (0.5 + 0.5 * n);
}

interface Fixture {
  base: number;
  phase: number;
  /** Emissive material of the square light panel (flickered with the point light). */
  panel?: THREE.MeshStandardMaterial;
}

/** Per-entity overhead fixtures: the point lights (illumination) plus the square
 * ceiling panels (the visible light). */
export interface EntityLights {
  lights: THREE.PointLight[];
  panels: THREE.Mesh[];
}

export class BackroomsScene {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly verticalFovDeg: number;

  private readonly config: EngineConfig;
  private readonly fixtures = new Map<THREE.PointLight, Fixture>();
  private flickerTime = 0;
  private readonly tmpVec = new THREE.Vector3();

  constructor(config: EngineConfig) {
    this.config = config;
    this.scene = new THREE.Scene();

    const aspect = config.renderWidth / config.renderHeight;
    this.verticalFovDeg = verticalFovDeg(config.fovHorizontalDeg, aspect);
    // Near plane kept small so a wall the player is walking right next to (down
    // to a few centimetres) is never clipped by the frustum — a near plane of
    // 0.1 made close walls vanish, opening a see-through hole to geometry (and
    // second-degree blackout quads) behind them. Far stays generous; fog (8..40)
    // hides any depth-precision loss beyond it.
    this.camera = new THREE.PerspectiveCamera(this.verticalFovDeg, aspect, 0.02, 200);

    const fogColor = new THREE.Color(config.fogColor);
    this.scene.fog = new THREE.Fog(fogColor, config.fogNear, config.fogFar);
    this.scene.background = fogColor.clone();

    // Ambient floor so corners are never pitch black (only the 5.5 blackout
    // quads are allowed to be pure black).
    this.scene.add(new THREE.AmbientLight(0xffffff, config.ambientLightIntensity));
  }

  /** Set the camera pose from the player: fixed eye height, horizontal forward
   * = (cos heading, 0, sin heading), up = +Y. Must run before projecting or
   * rendering. */
  positionCamera(player: PlayerState): void {
    const eye = this.config.eyeHeight;
    const fx = Math.cos(player.heading);
    const fz = Math.sin(player.heading);
    this.camera.position.set(player.x, eye, player.z);
    this.camera.lookAt(player.x + fx, eye, player.z + fz);
    this.camera.updateMatrixWorld(true);
  }

  /** Project a world point to normalized device screen X. Returns null when
   * the point is at or behind the camera plane. The returned X may lie slightly
   * outside [-1, 1] for points still in front but outside the horizontal
   * frustum — callers that need a clamped value should clamp it themselves. */
  projectToScreenX(wx: number, wz: number, wy: number): number | null {
    const camera = this.camera;
    camera.updateMatrixWorld(true);
    const v = this.tmpVec.set(wx, wy, wz);
    v.applyMatrix4(camera.matrixWorldInverse);
    if (v.z >= -1e-9) return null; // camera looks down its local -Z
    v.applyMatrix4(camera.projectionMatrix); // Vector3.applyMatrix4 divides by w
    return v.x;
  }

  /** Create and add overhead fixture lights covering one entity footprint. */
  addEntityLights(descriptor: EntityDescriptor): EntityLights {
    const lights: THREE.PointLight[] = [];
    const panels: THREE.Mesh[] = [];
    const ceilY = descriptor.kind === 'room'
      ? (descriptor.ceilingHeight ?? this.config.hallwayHeight)
      : this.config.hallwayHeight;
    for (const pos of fixturePositions(descriptor, this.config)) {
      const light = new THREE.PointLight(
        0xffffff,
        this.config.fixtureLightIntensity,
        0,
        LIGHT_DECAY,
      );
      light.position.copy(pos);
      this.scene.add(light);

      // Square light panel flush against the ceiling, centered on the same tile
      // as the point light (pos.z/x are already snapped to tile centers).
      const panel = makeCeilingPanel(pos.x, pos.z, ceilY - 0.02);
      this.scene.add(panel);
      panels.push(panel);

      this.fixtures.set(light, {
        base: this.config.fixtureLightIntensity,
        phase: fixturePhase(this.fixtures.size),
        panel: panel.material as THREE.MeshStandardMaterial,
      });
      lights.push(light);
    }
    this.applyFlicker();
    return { lights, panels };
  }

  /** Remove the given fixture lights (created by addEntityLights). */
  removeEntityLights(entityLights: EntityLights): void {
    for (const light of entityLights.lights) {
      this.scene.remove(light);
      this.fixtures.delete(light);
    }
    for (const panel of entityLights.panels) {
      this.scene.remove(panel);
      panel.geometry.dispose();
      (panel.material as THREE.Material).dispose();
    }
  }

  /** Advance the flicker clock by dt simulated seconds and re-apply. */
  updateFlicker(dt: number): void {
    this.flickerTime += dt;
    this.applyFlicker();
  }

  dispose(): void {
    for (const light of this.fixtures.keys()) {
      this.scene.remove(light);
    }
    this.fixtures.clear();
  }

  private applyFlicker(): void {
    for (const [light, fixture] of this.fixtures) {
      const mult = flickerMultiplier(this.config, fixture.phase, this.flickerTime);
      light.intensity = fixture.base * mult;
      if (fixture.panel) fixture.panel.emissiveIntensity = PANEL_EMISSIVE * mult;
    }
  }
}

/** Grid of ceiling fixture positions covering an entity footprint. Every
 * position is snapped to a ceiling-tile center (the same grid the ceiling
 * texture is centered on in geometry.ts) so each light sits dead-center in one
 * tile. */
function fixturePositions(
  descriptor: EntityDescriptor,
  config: EngineConfig,
): THREE.Vector3[] {
  const spacing = config.lightSpacing;
  const positions: THREE.Vector3[] = [];
  const { ax, az } = ceilingAnchors(descriptor);

  if (descriptor.kind === 'hallway' && descriptor.axis && descriptor.start && descriptor.end) {
    const ceilY = config.hallwayHeight - LIGHT_DROP;
    if (descriptor.axis === 'x') {
      const x0 = Math.min(descriptor.start.x, descriptor.end.x);
      const x1 = Math.max(descriptor.start.x, descriptor.end.x);
      // Snap the first fixture to a tile center; spacing is integer so every
      // subsequent fixture (x += spacing) stays on a tile center too.
      for (let x = snapToCeilingTile(x0 + spacing / 2, ax); x <= x1; x += spacing) {
        positions.push(new THREE.Vector3(x, ceilY, az));
      }
    } else {
      const z0 = Math.min(descriptor.start.z, descriptor.end.z);
      const z1 = Math.max(descriptor.start.z, descriptor.end.z);
      for (let z = snapToCeilingTile(z0 + spacing / 2, az); z <= z1; z += spacing) {
        positions.push(new THREE.Vector3(ax, ceilY, z));
      }
    }
    return positions;
  }

  if (descriptor.kind === 'room' && descriptor.center && descriptor.width && descriptor.depth) {
    const ceilY = (descriptor.ceilingHeight ?? config.hallwayHeight) - LIGHT_DROP;
    const cols = Math.max(1, Math.round(descriptor.width / spacing));
    const rows = Math.max(1, Math.round(descriptor.depth / spacing));
    for (let i = 0; i < cols; i++) {
      const x = descriptor.center.x - descriptor.width / 2 + ((i + 0.5) * descriptor.width) / cols;
      for (let j = 0; j < rows; j++) {
        const z = descriptor.center.z - descriptor.depth / 2 + ((j + 0.5) * descriptor.depth) / rows;
        positions.push(
          new THREE.Vector3(snapToCeilingTile(x, ax), ceilY, snapToCeilingTile(z, az)),
        );
      }
    }
    return positions;
  }

  return positions;
}

/** A bright square ceiling panel (recessed light) at (x, z), just below ceiling
 * height y, sized to sit inside a single ceiling tile. */
function makeCeilingPanel(x: number, z: number, y: number): THREE.Mesh {
  const geometry = new THREE.PlaneGeometry(PANEL_SIZE, PANEL_SIZE);
  const material = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    emissive: 0xffffff,
    emissiveIntensity: PANEL_EMISSIVE,
    roughness: 0.9,
    metalness: 0,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.rotation.x = Math.PI / 2; // face down (-Y) toward the player
  mesh.position.set(x, y, z);
  return mesh;
}
