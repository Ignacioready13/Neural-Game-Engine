import type { EngineConfig } from '../../config/config';
import type { CandidateDoorway, PlayerState, Vec2 } from '../core/types';
import { angleDiff, clamp, wrapAngle } from '../core/math';
import { resolveTarget } from './doorwayResolution';

/**
 * Steering: point-and-click navigation control (Section 6).
 *
 * Pure, deterministic kinematics — no timers, no Math.random. The player always
 * walks forward at a fixed speed; the only thing input changes is WHICH
 * candidate doorway the turn is steering toward. This exact class is shared by
 * human-interactive mode and headless bot mode (Section 9.3), so the training
 * signal is statistically identical in both.
 *
 * Documented judgment calls (see root README):
 *  - `desiredOmega` is a proportional controller toward the target heading,
 *      omega = clamp(turnRateGain * err, +-maxTurnRate), then the change in
 *      omega is limited by the jerk threshold. A P-controller converges
 *      monotonically (heading error decays without overshoot), so motion is
 *      smooth and never oscillates — unlike a bang-bang minimum-time profile,
 *      which overshoots in discrete time and reads as horizontal camera jitter.
 *  - The steering TARGET is sticky: it is resolved once per accepted click
 *    (nearest reachable doorway to the click's screen X at that instant) and
 *    held across frames until it leaves the candidate field or a new accepted
 *    click arrives. It is NOT re-derived from the held click X each frame —
 *    screen positions move as the player turns, so re-deriving would make the
 *    target flap between doorways mid-turn.
 *  - `isReachable` is an approximate, conservative geometric feasibility test
 *    (documented on the method): a candidate is reachable only if it is inside
 *    the front cone AND the straight-line distance to it is at least the arc
 *    length needed to finish the turn. It is hand-tunable via config.
 */
export class Steering {
  private readonly cfg: EngineConfig;

  private _lastClickedX: number | null = null;
  private _targetDoorwayId: string | null = null;
  private _lastClickTime = 0;
  private _lastClickScreenX: number | null = null;
  private _lastAcceptedTime = Number.NEGATIVE_INFINITY;
  /** Set true by an accepted click so the next update() resolves a fresh target. */
  private _pendingResolution = false;

  constructor(config: EngineConfig) {
    this.cfg = config;
  }

  /** Held control signal (Section 6.1), normalized -1..1. Null before the
   * first accepted click; only ever changes on an ACCEPTED click. */
  get lastClickedX(): number | null {
    return this._lastClickedX;
  }

  /** Currently selected target doorway id; null when none yet. */
  get targetDoorwayId(): string | null {
    return this._targetDoorwayId;
  }

  /** Simulated time of the last click of ANY kind (for the pulse UI). */
  get lastClickTime(): number {
    return this._lastClickTime;
  }

  /** Screen X of the last click of any kind (for the pulse UI). */
  get lastClickScreenX(): number | null {
    return this._lastClickScreenX;
  }

  /**
   * Feed a new click (normalized screen X, -1..1; Y is ignored per spec).
   * Every click is recorded for the click pulse; the debounce only gates
   * whether the click also re-targets the held control signal.
   */
  handleClick(screenX: number, simTime: number): { accepted: boolean } {
    const x = clamp(screenX, -1, 1);
    this._lastClickTime = simTime;
    this._lastClickScreenX = x;

    const accepted = simTime - this._lastAcceptedTime >= this.cfg.debounceSeconds;
    if (accepted) {
      this._lastClickedX = x;
      this._lastAcceptedTime = simTime;
      this._pendingResolution = true;
    }
    return { accepted };
  }

  /**
   * Compute the current target from live candidates each frame.
   *
   * - On the frame after an accepted click, resolve a fresh sticky target from
   *   the new held click X (nearest reachable doorway at that instant).
   * - With no click ever, drift toward the reachable doorway nearest dead
   *   center, re-evaluated each frame (Section 6.6).
   * - With a held click and no new click, keep the sticky target while that
   *   doorway is still a LIVE doorway of the active entity; only re-resolve
   *   once it is genuinely gone (the player walked through it into the
   *   neighbor, so it is no longer a doorway of the entity they now occupy).
   *
   * `liveDoorwayIds` is every doorway id of the active entity, regardless of
   * whether each is currently projectable/reachable. Holding against THIS set
   * (rather than the projectable `candidates`) is what makes a committed turn
   * follow through: a doorway's entry point goes behind the camera mid-turn,
   * which would drop it from `candidates` and — if we keyed stickiness on
   * `candidates` — would re-resolve the target and turn the player away from
   * the doorway they just selected.
   */
  update(
    candidates: CandidateDoorway[],
    liveDoorwayIds: ReadonlyArray<string>,
    _simTime: number,
  ): string | null {
    if (this._pendingResolution) {
      this._targetDoorwayId = resolveTarget(
        candidates,
        this._lastClickedX,
        this._targetDoorwayId,
      );
      this._pendingResolution = false;
      return this._targetDoorwayId;
    }

    if (this._lastClickedX === null) {
      // Default/idle: always drift toward whatever is straightest ahead.
      this._targetDoorwayId = resolveTarget(candidates, null, this._targetDoorwayId);
      return this._targetDoorwayId;
    }

    // Held click, no new click: hold the sticky target while it is still a
    // live doorway of the active entity.
    if (
      this._targetDoorwayId !== null &&
      liveDoorwayIds.includes(this._targetDoorwayId)
    ) {
      return this._targetDoorwayId;
    }

    // The sticky target is genuinely gone (the player entered its neighbor).
    this._targetDoorwayId = resolveTarget(candidates, this._lastClickedX, null);
    return this._targetDoorwayId;
  }

  /**
   * Advance the player one timestep toward `target` (null = keep straight).
   * Kinematic steering with a jerk (angular-acceleration) limit on omega:
   * heading, angular velocity, and position are all continuous across calls
   * and across retargets.
   */
  step(player: PlayerState, target: Vec2 | null, dt: number): PlayerState {
    const desiredHeading =
      target === null
        ? player.heading
        : Math.atan2(target.z - player.z, target.x - player.x);

    const headingError = angleDiff(player.heading, desiredHeading);

    // Proportional controller: turn rate grows with heading error, capped at
    // maxTurnRate. Monotonic convergence -> no overshoot/oscillation.
    const desiredOmega = clamp(
      this.cfg.turnRateGain * headingError,
      -this.cfg.maxTurnRate,
      this.cfg.maxTurnRate,
    );

    const jerkLimit = this.cfg.jerkThreshold * dt;
    let omega = clamp(
      player.angularVelocity +
        clamp(desiredOmega - player.angularVelocity, -jerkLimit, jerkLimit),
      -this.cfg.maxTurnRate,
      this.cfg.maxTurnRate,
    );

    const heading = wrapAngle(player.heading + omega * dt);
    const x = player.x + Math.cos(heading) * this.cfg.walkSpeed * dt;
    const z = player.z + Math.sin(heading) * this.cfg.walkSpeed * dt;

    return { x, z, heading, angularVelocity: omega };
  }

  /**
   * Feasibility: can a jerk-limited turn from `player` plausibly reach `entry`
   * before overshooting?
   *
   * Conservative approximation (documented, hand-tunable):
   *  (a) the entry must be in front — the angle to it must be within
   *      reachabilityMaxAngleDeg of the heading; and
   *  (b) the straight-line distance to the entry must be at least the arc
   *      length the player travels while completing the turn, estimated as the
   *      minimum-time bang-bang turn (triangle/trapezoid in omega, capped at
   *      maxTurnRate) plus a deceleration allowance for the current omega.
   *
   * The straight-line distance is a lower bound on the curved path length, so
   * this check can only under-report reachability, never over-report it. A
   * null/NaN/Infinity entry is treated as unreachable.
   */
  isReachable(player: PlayerState, entry: Vec2): boolean {
    if (!entry || !Number.isFinite(entry.x) || !Number.isFinite(entry.z)) {
      return false;
    }
    const dx = entry.x - player.x;
    const dz = entry.z - player.z;
    const dist = Math.hypot(dx, dz);
    if (!(dist > 0)) return false;

    const headingError = Math.abs(
      angleDiff(player.heading, Math.atan2(dz, dx)),
    );
    const maxAngleRad = (this.cfg.reachabilityMaxAngleDeg * Math.PI) / 180;
    if (headingError > maxAngleRad) return false;

    const required = this.requiredArcLength(
      headingError,
      Math.abs(player.angularVelocity),
    );
    return Number.isFinite(required) && dist >= required;
  }

  /**
   * Forward distance the player travels while rotating `angle` radians to a
   * stop from an initial angular speed `omega0`, at the jerk limit. Returns
   * Infinity when the initial spin alone would overshoot the target heading.
   */
  private requiredArcLength(angle: number, omega0: number): number {
    const j = this.cfg.jerkThreshold;
    const wMax = this.cfg.maxTurnRate;
    const v = this.cfg.walkSpeed;

    // Heading swept while braking from omega0 at max jerk: omega0^2 / (2j).
    // If that already passes the target heading, we would overshoot.
    const stopAngle = (omega0 * omega0) / (2 * j);
    if (stopAngle > angle) return Number.POSITIVE_INFINITY;

    // Minimum time to rotate `angle` and come to rest from omega = 0.
    // Triangle profile (peak below wMax) vs trapezoid (capped at wMax).
    const cruiseAngle = (wMax * wMax) / j;
    let t: number;
    if (angle <= cruiseAngle) {
      t = 2 * Math.sqrt(angle / j);
    } else {
      t = (2 * wMax) / j + (angle - cruiseAngle) / wMax;
    }

    // Conservative allowance for a non-zero initial omega (the true optimal
    // profile starts from omega0 rather than rest, so this only overestimates).
    t += omega0 / j;

    return v * t;
  }
}
