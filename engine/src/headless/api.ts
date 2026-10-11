/**
 * SHARED HEADLESS API TYPES
 * =========================
 * The `window.__backrooms` contract, shared between the browser entry
 * (src/app/main-headless.ts, which implements it) and the Node driver
 * (src/headless/driver.ts, which calls it over the Puppeteer bridge).
 *
 * The `declare global` block augments the DOM `Window` type so both sides can
 * reference `window.__backrooms` / `window.__BACKROOMS_SEED__` without casts.
 */
import type { PlayerState } from '../core/types';

export interface GpuInfo {
  renderer: string;
  vendor: string;
}

export interface FrameCapture {
  dataUrl: string;
  width: number;
  height: number;
}

export interface BackroomsState {
  player: PlayerState;
  activeEntityId: string;
  frameIndex: number;
  simulatedTime: number;
  /** The held control signal in effect for the CURRENT state (-1..1; 0 before
   * any click). Recorded verbatim in the manifest. */
  controlSignalX: number;
  /** True once the player has ended up outside every floor (wall/void overlap). */
  wallOverlap?: boolean;
  /** Count of collision-slide events (a move blocked/redirected by a wall). */
  collisionEvents?: number;
  /** Raw last accepted click X (null before any click) — debug only. */
  lastClickedX?: number | null;
  /** True once at least one click has ever been accepted (false = "never
   * clicked" default mode). */
  hasAcceptedClick?: boolean;
  /** Acceptance result of the click submitted THIS step. */
  clickAccepted?: boolean | null;
  /** True if the wall-slide backstop engaged THIS step. */
  slidThisStep?: boolean;
  /** Currently selected target doorway id (null when none). */
  targetDoorwayId?: string | null;
  /** Current steering target point (world space), or null when none. */
  targetPoint?: { x: number; z: number } | null;
  /** Which leg of the doorway approach the player is on. */
  targetPhase?: 'approach' | 'through';
  /** This frame's candidate doorways (id + screen X + reachability). */
  candidates?: Array<{ doorwayId: string; screenX: number; reachable: boolean; screenDistance: number }>;
}

export interface BackroomsApi {
  /** Idempotent; ensures the engine exists and has rendered once. */
  init(): Promise<void>;
  /** `clickX` = a NEW click this frame (normalized -1..1) or null. Feeds
   * submitClick (if any) then advances exactly one fixed simulated timestep. */
  step(clickX: number | null): void;
  /** Render the latest state and return the OUTPUT-resolution PNG data URL. */
  captureFrame(): FrameCapture;
  getState(): BackroomsState;
  /** WebGL UNMASKED_RENDERER/VENDOR via WEBGL_debug_renderer_info. Queryable
   * BEFORE init (uses a probe canvas). */
  gpuInfo(): GpuInfo;
}

declare global {
  interface Window {
    __backrooms: BackroomsApi;
    /** Episode seed injected by the driver before the page scripts run. */
    __BACKROOMS_SEED__?: number;
  }
}
