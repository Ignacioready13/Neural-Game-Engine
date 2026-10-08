/**
 * Lazy world generation + disposal.
 *
 * The World owns a single seeded PRNG, stable entity ids, and the map of live
 * entities. Only the active entity and its direct (generated) neighbors exist
 * at any time: maintain() generates the active entity's not-yet-generated
 * doorway neighbors, then disposes everything that is neither active nor a
 * direct neighbor. There is no global floor plan — the world is discovered
 * lazily and deterministically.
 */
import * as THREE from 'three';
import type { Doorway, Entity, EntityDescriptor } from '../core/types';
import type { EngineConfig } from '../../config/config';
import type { Rng } from '../core/rng';
import { createRng } from '../core/rng';
import { makeIds, Ids } from './doorway';
import { generateStartHallway, generateNeighbor, containsPoint, GeneratedChild } from './layout';
import { buildEntityGroup } from './geometry';

export interface WorldOptions {
  /** Build three.js geometry (default true). Set false in plain-Node tests. */
  buildGeometry?: boolean;
}

export class World {
  private readonly rng: Rng;
  private readonly config: EngineConfig;
  private readonly ids: Ids = makeIds();
  private readonly entities = new Map<string, Entity>();
  private readonly buildGeom: boolean;

  constructor(seed: number, config: EngineConfig, opts: WorldOptions = {}) {
    this.rng = createRng(seed);
    this.config = config;
    this.buildGeom = opts.buildGeometry ?? true;
  }

  /** Create the starting hallway (no entry doorway) and return it. */
  init(): Entity {
    const child = generateStartHallway(this.ids, this.rng, this.config);
    const entity = this.instantiate(child);
    this.entities.set(child.descriptor.id, entity);
    return entity;
  }

  private instantiate(child: GeneratedChild): Entity {
    const group = this.buildGeom
      ? buildEntityGroup(child.descriptor, this.config, {
          skipEntryWall: child.skipEntryWall,
          parentDir: child.parentDir,
        })
      : null;
    return { descriptor: child.descriptor, group };
  }

  /**
   * Generate every not-yet-generated neighbor of `activeId`, then dispose every
   * entity that is neither active nor a direct neighbor. Returns what changed.
   */
  maintain(activeId: string): { generated: Entity[]; disposed: string[] } {
    const active = this.entities.get(activeId);
    if (!active) return { generated: [], disposed: [] };

    const generated: Entity[] = [];
    for (const doorway of active.descriptor.doorways) {
      if (doorway.targetGenerated) continue;
      const child = generateNeighbor(this.ids, this.rng, this.config, active.descriptor, doorway);
      doorway.targetEntityId = child.descriptor.id;
      doorway.targetGenerated = true;
      const entity = this.instantiate(child);
      this.entities.set(child.descriptor.id, entity);
      generated.push(entity);
    }

    const alive = new Set<string>([activeId]);
    for (const doorway of active.descriptor.doorways) {
      if (doorway.targetGenerated && doorway.targetEntityId) alive.add(doorway.targetEntityId);
    }

    const disposed: string[] = [];
    for (const [id, entity] of this.entities) {
      if (alive.has(id)) continue;
      this.disposeEntity(entity);
      this.entities.delete(id);
      disposed.push(id);
    }

    // Any live entity's doorway that pointed at a just-disposed entity must no
    // longer reference it. Non-'start' doorways become ungenerated so they
    // regenerate lazily when that entity is re-entered. A hallway's 'start'
    // doorway is the entry back to its parent: once that parent is disposed
    // there is no valid neighbor to regenerate on the other side, so it becomes
    // a permanent dead-end (targetGenerated stays true, target is cleared) and
    // is never rebuilt into a spurious backwards hallway.
    if (disposed.length > 0) {
      const disposedSet = new Set(disposed);
      for (const entity of this.entities.values()) {
        for (const d of entity.descriptor.doorways) {
          if (!d.targetGenerated || !disposedSet.has(d.targetEntityId)) continue;
          d.targetEntityId = '';
          d.targetGenerated = d.side === 'start';
        }
      }
    }
    return { generated, disposed };
  }

  get(id: string): Entity | undefined {
    return this.entities.get(id);
  }

  /** All currently-live entities. */
  all(): Entity[] {
    return Array.from(this.entities.values());
  }

  /**
   * Doorways of the active entity's generated neighbors whose own doorways lead
   * to not-yet-generated entities (second-degree => opaque black fill).
   */
  blackDoorways(activeId: string): Doorway[] {
    const active = this.entities.get(activeId);
    if (!active) return [];
    const result: Doorway[] = [];
    for (const d of active.descriptor.doorways) {
      if (!d.targetGenerated || !d.targetEntityId) continue;
      const neighbor = this.entities.get(d.targetEntityId);
      if (!neighbor) continue;
      for (const nd of neighbor.descriptor.doorways) {
        if (!nd.targetGenerated) result.push(nd);
      }
    }
    return result;
  }

  /** The entity whose interior contains a floor point (for enter/exit tests). */
  entityAt(x: number, z: number): EntityDescriptor | undefined {
    for (const entity of this.entities.values()) {
      if (containsPoint(entity.descriptor, x, z, this.config)) return entity.descriptor;
    }
    return undefined;
  }

  /** True when the floor point lies inside ANY live entity's floor. Used by the
   * collision check: a point outside every floor is inside a wall or the void,
   * so the player must not be allowed there. */
  containsFloorPoint(x: number, z: number): boolean {
    for (const entity of this.entities.values()) {
      if (containsPoint(entity.descriptor, x, z, this.config)) return true;
    }
    return false;
  }

  /** True when the floor point lies inside the entity `id`'s floor (false when
   * that entity is no longer live). Used for hysteresis in entity tracking. */
  containsEntity(id: string, x: number, z: number): boolean {
    const entity = this.entities.get(id);
    return entity ? containsPoint(entity.descriptor, x, z, this.config) : false;
  }

  private disposeEntity(entity: Entity): void {
    const group = entity.group as THREE.Group | null;
    if (!group) return;
    group.traverse((obj) => {
      const anyObj = obj as unknown as {
        isMesh?: boolean;
        geometry?: THREE.BufferGeometry;
        material?: THREE.Material | THREE.Material[];
      };
      if (!anyObj.isMesh) return;
      if (anyObj.geometry) anyObj.geometry.dispose();
      const mat = anyObj.material;
      if (Array.isArray(mat)) {
        mat.forEach((m) => m.dispose());
      } else if (mat) {
        mat.dispose();
      }
    });
    if (group.parent) group.parent.remove(group);
  }
}
