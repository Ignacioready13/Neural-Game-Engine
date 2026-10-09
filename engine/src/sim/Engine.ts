/**
 * SHARED CORE ENGINE
 * ==================
 * This is the single orchestrator used IDENTICALLY by human-interactive mode
 * and headless bot mode (Section 9.3). It wires together the world generator,
 * the steering/control logic, and the renderer, and advances one fixed
 * simulated timestep per `step()` call.
 *
 * The two modes differ only in:
 *   - where input comes from: real mouse (main-interactive) vs. the bot
 *     feeding `submitClick` (main-headless),
 *   - where output goes: a visible canvas vs. `captureFrame()` -> disk.
 *
 * The world generation, control/spline/jerk, and fixed-timestep logic live in
 * the shared modules below and are never duplicated.
 */
import type { EngineConfig } from '../../config/config';
import { CARDINAL_VEC } from '../core/types';
import type {
  CandidateDoorway,
  Doorway,
  Entity,
  EntityDescriptor,
  FrameSnapshot,
  PlayerState,
  SteeringState,
  Vec2,
} from '../core/types';
import { World } from '../world/World';
import { Steering } from '../control/steering';
import { Renderer } from '../render/Renderer';
import { resolveWallCollision } from './collision';
import { doorwayApproachPoint, doorwayPhase, doorwayTargetPoint } from './navigation';
import type { DoorwayPhase } from './navigation';

interface Pulse {
  screenX: number;
  ageSeconds: number;
}

export class Engine {
  readonly config: EngineConfig;
  readonly seed: number;

  private readonly world: World;
  private readonly steering: Steering;
  private readonly renderer: Renderer;

  private player: PlayerState;
  private activeEntityId: string;
  private frameIndex = 0;
  private simulatedTime = 0;

  private pendingClickX: number | null = null;
  private pulses: Pulse[] = [];
  private blackDoorways: Doorway[] = [];
  private currentTargetPoint: Vec2 | null = null;
  private targetDoorwayId: string | null = null;
  private targetPhase: DoorwayPhase = 'approach';
  private candidates: CandidateDoorway[] = [];

  // Geometry-overlap diagnostics (exposed via getState for investigation).
  private wallOverlap = false;
  private wallOverlapCount = 0;
  private lastOverlapWarnTime = Number.NEGATIVE_INFINITY;
  private collisionEvents = 0;

  constructor(config: EngineConfig, seed: number, canvas: HTMLCanvasElement) {
    this.config = config;
    this.seed = seed;

    this.world = new World(seed, config);
    this.steering = new Steering(config);
    this.renderer = new Renderer(canvas, config);

    // 5.6: start partway down a short starting hallway, facing down its length.
    const start = this.world.init();
    this.activeEntityId = start.descriptor.id;
    this.syncWorld();
    this.player = this.startingPlayer(start.descriptor);
    this.currentTargetPoint = null;

    this.renderer.positionCamera(this.player);
  }

  /** Feed a click (normalized screen X, -1..1) to be processed on the next step. */
  submitClick(screenX: number): void {
    this.pendingClickX = screenX;
  }

  /** Advance exactly one fixed simulated timestep. */
  step(): void {
    const dt = this.config.simTimestep;

    // Process any new click (debounce + pulse handled in Steering).
    if (this.pendingClickX !== null) {
      this.steering.handleClick(this.pendingClickX, this.simulatedTime);
      this.pulses.push({ screenX: this.pendingClickX, ageSeconds: 0 });
      this.pendingClickX = null;
    }

    // Collect candidates = active entity's doorways, projected to screen X.
    const active = this.world.get(this.activeEntityId);
    this.renderer.positionCamera(this.player);
    this.candidates = this.collectCandidates(active);
    const liveDoorwayIds = active ? active.descriptor.doorways.map((d) => d.id) : [];

    // Resolve the steering target (default = screen center when no click held).
    const targetId = this.steering.update(this.candidates, liveDoorwayIds, this.simulatedTime);
    if (targetId !== this.targetDoorwayId) {
      // Restart the approach/through cycle whenever the target doorway changes.
      this.targetDoorwayId = targetId;
      this.targetPhase = 'approach';
    }
    const targetDoorway = this.findDoorway(active, targetId);
    if (targetDoorway) {
      this.targetPhase = doorwayPhase(this.player, targetDoorway, this.config, this.targetPhase);
      this.currentTargetPoint = doorwayTargetPoint(targetDoorway, this.targetPhase);
    } else {
      this.currentTargetPoint = null;
      this.targetPhase = 'approach';
    }

    // Advance the player one timestep along the spline/jerk-limited path, then
    // resolve any wall collision (slide along the wall instead of clipping).
    const stepped = this.steering.step(this.player, this.currentTargetPoint, dt);
    this.player = resolveWallCollision(this.world, this.player, stepped, () => {
      this.collisionEvents += 1;
    });

    // Entity tracking with HYSTERESIS: only leave the active entity once it no
    // longer contains the player. This is the robust ground truth the player is
    // "always in the entity we think they are in": a boundary graze or a botched
    // clip can never flap the active entity (and therefore never churn the
    // second-degree blackout set) — the player only hands off when they have
    // genuinely left the current entity's floor.
    const stillInside = this.world.containsEntity(this.activeEntityId, this.player.x, this.player.z);
    if (!stillInside) {
      const here = this.world.entityAt(this.player.x, this.player.z);
      if (here && here.id !== this.activeEntityId) {
        this.activeEntityId = here.id;
        this.syncWorld();
      }
    }

    // Geometry-overlap checker: flag/warn if the player ended up outside every
    // floor (inside a wall or the void) — should never happen once collision is
    // in place, but this surfaces any remaining geometry gap for investigation.
    if (!this.world.containsFloorPoint(this.player.x, this.player.z)) {
      this.wallOverlap = true;
      this.wallOverlapCount += 1;
      if (this.simulatedTime - this.lastOverlapWarnTime >= 1.0) {
        this.lastOverlapWarnTime = this.simulatedTime;
        console.warn(
          `[overlap] player at (${this.player.x.toFixed(2)}, ${this.player.z.toFixed(2)}) ` +
            `is inside a wall or the void (active=${this.activeEntityId}, frame=${this.frameIndex})`,
        );
      }
    }

    this.renderer.updateFlicker(dt);
    this.simulatedTime += dt;
    this.frameIndex += 1;

    // Age and prune click pulses.
    for (const p of this.pulses) p.ageSeconds += dt;
    this.pulses = this.pulses.filter((p) => p.ageSeconds <= this.config.clickPulseSeconds);
  }

  /** Render the current state (supersample -> downsample -> screen). */
  render(): void {
    this.renderer.render(this.snapshot());
  }

  /** Return the final OUTPUT-resolution frame as a PNG data URL. */
  captureFrame(): { dataUrl: string; width: number; height: number } {
    this.renderer.render(this.snapshot());
    return this.renderer.captureFrame();
  }

  getState(): {
    player: PlayerState;
    activeEntityId: string;
    frameIndex: number;
    simulatedTime: number;
    /** The held control signal in effect for the CURRENT state (last accepted
     * click X, -1..1; 0 = screen-center default before any click). This is the
     * value recorded in the manifest. */
    controlSignalX: number;
    /** True once the player has ever ended up outside every floor (a wall/void
     * overlap) — the geometry-overlap checker's signal. */
    wallOverlap: boolean;
    /** How many times the collision slide blocked/redirected a move. */
    collisionEvents: number;
    /** Raw last accepted click X (null before any click) — debug only. */
    lastClickedX: number | null;
    /** Currently selected target doorway id (null when none). */
    targetDoorwayId: string | null;
    /** Current steering target point (world space), or null when none. */
    targetPoint: Vec2 | null;
    /** Which leg of the doorway approach the player is on. */
    targetPhase: DoorwayPhase;
    /** This frame's candidate doorways (id + screen X + reachability). */
    candidates: Array<{ doorwayId: string; screenX: number; reachable: boolean; screenDistance: number }>;
  } {
    return {
      player: this.player,
      activeEntityId: this.activeEntityId,
      frameIndex: this.frameIndex,
      simulatedTime: this.simulatedTime,
      controlSignalX: this.steering.lastClickedX ?? 0,
      wallOverlap: this.wallOverlap,
      collisionEvents: this.collisionEvents,
      lastClickedX: this.steering.lastClickedX,
      targetDoorwayId: this.targetDoorwayId,
      targetPoint: this.currentTargetPoint ? { x: this.currentTargetPoint.x, z: this.currentTargetPoint.z } : null,
      targetPhase: this.targetPhase,
      candidates: this.candidates.map((c) => ({
        doorwayId: c.doorwayId,
        screenX: c.screenX,
        reachable: c.reachable,
        screenDistance: c.screenDistance,
      })),
    };
  }

  get playerState(): PlayerState {
    return this.player;
  }

  get activeEntity(): EntityDescriptor | undefined {
    return this.world.get(this.activeEntityId)?.descriptor;
  }

  // -------------------------------------------------------------------------

  private startingPlayer(start: EntityDescriptor): PlayerState {
    const dir = start.travelDirection ? CARDINAL_VEC[start.travelDirection] : { x: 1, z: 0 };
    const startPt = start.start ?? { x: 0, z: 0 };
    const inset = Math.min(1.5, this.config.hallwayLengthMin / 2);
    return {
      x: startPt.x + dir.x * inset,
      z: startPt.z + dir.z * inset,
      heading: Math.atan2(dir.z, dir.x),
      angularVelocity: 0,
    };
  }

  private collectCandidates(active: Entity | undefined): CandidateDoorway[] {
    if (!active) return [];
    const out: CandidateDoorway[] = [];
    const reference = this.steering.lastClickedX ?? 0;
    for (const d of active.descriptor.doorways) {
      // Candidates are evaluated against the doorway's APPROACH point (the
      // corner for a bend, the opening center otherwise) — the point the player
      // steers toward on the approach leg. Evaluating against `entryPoint` is
      // wrong for a bend: its entryPoint is a waypoint past the corner, which
      // sits far to the side and can falsely fail the forward reachability cone.
      const approach = doorwayApproachPoint(d);
      const sx = this.renderer.projectToScreenX(
        approach.x,
        approach.z,
        this.config.eyeHeight,
      );
      if (sx === null) continue;
      out.push({
        doorwayId: d.id,
        screenX: sx,
        reachable: this.steering.isReachable(this.player, approach),
        screenDistance: Math.abs(sx - reference),
      });
    }
    return out;
  }

  private findDoorway(active: Entity | undefined, id: string | null): Doorway | null {
    if (!active || id === null) return null;
    return active.descriptor.doorways.find((d) => d.id === id) ?? null;
  }

  /** Regenerate neighbors for the active entity and sync the renderer. */
  private syncWorld(): void {
    const { generated, disposed } = this.world.maintain(this.activeEntityId);
    void generated;
    void disposed;
    this.renderer.setEntities(this.world.all());
    this.blackDoorways = this.world.blackDoorways(this.activeEntityId);
    this.renderer.setBlackDoorways(this.blackDoorways);
  }

  private snapshot(): FrameSnapshot {
    const steering: SteeringState = {
      lastClickedX: this.steering.lastClickedX,
      targetDoorwayId: this.steering.targetDoorwayId,
      lastClickTime: this.steering.lastClickTime,
      lastClickScreenX: this.steering.lastClickScreenX,
    };
    return {
      entities: this.world.all(),
      activeEntity: this.world.get(this.activeEntityId)!.descriptor,
      blackDoorways: this.blackDoorways,
      player: this.player,
      steering,
      clickPulses: this.pulses.map((p) => ({ screenX: p.screenX, ageSeconds: p.ageSeconds })),
      simulatedTime: this.simulatedTime,
      frameIndex: this.frameIndex,
    };
  }
}
