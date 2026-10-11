/**
 * CLICK-BOT FRAMEWORK (Stage 2, Section 4.6)
 * ==========================================
 * Replaces the single placeholder bot with a small, deterministic persona
 * framework. A `Bot` receives the frame index and the same read-only state
 * snapshot the logger reads that frame, and returns a decision: the click X in
 * [-1,1] (or null), the persona that made the decision, and the doorway the bot
 * aimed at (or null).
 *
 * Bots run in Node and keep their own memory (the active entity id, the current
 * segment persona, pending delayed clicks). Bot randomness comes from a seeded
 * generator DERIVED from the episode seed but entirely separate from the
 * world's generator, so bot decisions never perturb world-generation draw order.
 */
import type { EngineConfig } from '../../config/config';
import { CONFIG } from '../../config/config';
import { createRng, intRange } from '../core/rng';
import type { Rng } from '../core/rng';
import { clamp } from '../core/math';

export type PersonaId =
  | 'idle'
  | 'decisive'
  | 'late'
  | 'dither'
  | 'prober'
  | 'spammer'
  | 'uniform';

export type ProfileName = keyof EngineConfig['mixtureProfiles'];

/** Read-only state snapshot the bot reads each frame (subset of getState). */
export interface BotState {
  frameIndex: number;
  activeEntityId: string;
  player: { x: number; z: number; heading: number };
  candidates: Array<{ doorwayId: string; screenX: number; reachable: boolean; screenDistance: number }>;
  controlSignalX: number;
}

export interface BotDecision {
  clickX: number | null;
  persona: PersonaId;
  intentDoorwayId: string | null;
}

export type Bot = (frameIndex: number, state: BotState) => BotDecision;

/** Mix the episode seed into a distinct bot stream (independent of the world). */
function botSeed(seed: number): number {
  return (seed ^ 0x5f356495) >>> 0;
}

/** Standard-normal sample via Box-Muller from the bot's own rng. */
function gaussian(rng: Rng): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

class ClickBot {
  private readonly rng: Rng;
  private readonly cfg: EngineConfig;
  private readonly weights: Readonly<Record<string, number>>;

  // Segment scheduler.
  private persona: PersonaId;
  private segmentRemaining: number;

  // Choice-moment tracking (across the whole episode).
  private prevEntityId: string | null = null;
  private prevReachableIds = new Set<string>();

  // Per-persona state.
  private pending: { fireAt: number } | null = null; // decisive / late delayed click
  private ditherTarget: string | null = null;
  private ditherRethinkAt = 0;
  private spammerBurstRemaining = 0;
  private spammerNextAt = 0;
  private uniformNextAt = 0;

  constructor(seed: number, profile: ProfileName, cfg: EngineConfig) {
    this.rng = createRng(botSeed(seed));
    this.cfg = cfg;
    this.weights = cfg.mixtureProfiles[profile];
    this.persona = this.drawPersona();
    this.segmentRemaining = intRange(this.rng, cfg.botSegmentLengthMin, cfg.botSegmentLengthMax);
  }

  private drawPersona(): PersonaId {
    const entries = Object.entries(this.weights);
    const total = entries.reduce((s, [, w]) => s + w, 0);
    let roll = this.rng() * total;
    for (const [name, w] of entries) {
      roll -= w;
      if (roll <= 0) return name as PersonaId;
    }
    return entries[entries.length - 1]![0] as PersonaId;
  }

  private advanceSegment(): void {
    this.segmentRemaining -= 1;
    if (this.segmentRemaining <= 0) {
      this.persona = this.drawPersona();
      this.segmentRemaining = intRange(this.rng, this.cfg.botSegmentLengthMin, this.cfg.botSegmentLengthMax);
      this.pending = null;
      this.ditherTarget = null;
    }
  }

  decide(frameIndex: number, state: BotState): BotDecision {
    this.advanceSegment();
    const persona = this.persona;

    const reachable = state.candidates.filter((c) => c.reachable);
    const cur = new Set(reachable.map((c) => c.doorwayId));
    const entityEntry = this.prevEntityId !== null && state.activeEntityId !== this.prevEntityId;
    let gainedMember = false;
    for (const id of cur) if (!this.prevReachableIds.has(id)) { gainedMember = true; break; }
    this.prevReachableIds = cur;
    this.prevEntityId = state.activeEntityId;
    const triggered = entityEntry || gainedMember;

    let clickX: number | null = null;
    let intent: string | null = null;

    switch (persona) {
      case 'idle':
        break;

      case 'decisive':
        ({ clickX, intent } = this.delayedAim(frameIndex, reachable, triggered, this.cfg.botDecisiveDelayMin, this.cfg.botDecisiveDelayMax));
        break;

      case 'late':
        ({ clickX, intent } = this.delayedAim(frameIndex, reachable, triggered, this.cfg.botLateDelayMin, this.cfg.botLateDelayMax));
        break;

      case 'dither':
        ({ clickX, intent } = this.dither(frameIndex, reachable, triggered));
        break;

      case 'prober':
        if (triggered && this.pending === null) {
          this.pending = { fireAt: frameIndex };
        }
        if (this.pending && frameIndex >= this.pending.fireAt) {
          clickX = this.probePosition(reachable);
          this.pending = null;
        }
        break;

      case 'spammer':
        clickX = this.spammer(frameIndex);
        break;

      case 'uniform':
        if (this.uniformNextAt === 0) {
          this.uniformNextAt = frameIndex + intRange(this.rng, this.cfg.botUniformIntervalMin, this.cfg.botUniformIntervalMax);
        }
        if (frameIndex >= this.uniformNextAt) {
          clickX = this.rng() * 2 - 1;
          this.uniformNextAt = frameIndex + intRange(this.rng, this.cfg.botUniformIntervalMin, this.cfg.botUniformIntervalMax);
        }
        break;
    }

    return { clickX, persona, intentDoorwayId: intent };
  }

  /** Decisive/late: schedule one click per choice moment, fire after a delay. */
  private delayedAim(
    frameIndex: number,
    reachable: BotState['candidates'],
    triggered: boolean,
    delayMin: number,
    delayMax: number,
  ): { clickX: number | null; intent: string | null } {
    if (triggered && this.pending === null) {
      this.pending = { fireAt: frameIndex + intRange(this.rng, delayMin, delayMax) };
    }
    if (this.pending && frameIndex >= this.pending.fireAt) {
      this.pending = null;
      return this.aimAtRandomReachable(reachable);
    }
    return { clickX: null, intent: null };
  }

  /** Pick uniformly among reachable candidates, click at its screen X + jitter. */
  private aimAtRandomReachable(reachable: BotState['candidates']): { clickX: number | null; intent: string | null } {
    if (reachable.length === 0) return { clickX: null, intent: null };
    const pick = reachable[Math.floor(this.rng() * reachable.length)]!;
    const sigma = this.cfg.botJitterSigmaMin + this.rng() * (this.cfg.botJitterSigmaMax - this.cfg.botJitterSigmaMin);
    return { clickX: clamp(pick.screenX + gaussian(this.rng) * sigma, -1, 1), intent: pick.doorwayId };
  }

  /** Dither: choose a door, then re-click a DIFFERENT reachable door every so often. */
  private dither(
    frameIndex: number,
    reachable: BotState['candidates'],
    triggered: boolean,
  ): { clickX: number | null; intent: string | null } {
    if (this.ditherTarget !== null && !reachable.some((c) => c.doorwayId === this.ditherTarget)) {
      this.ditherTarget = null; // chosen door no longer reachable
    }
    if (this.ditherTarget === null) {
      if (!triggered) return { clickX: null, intent: null };
      const first = this.aimAtRandomReachable(reachable);
      if (first.intent === null) return { clickX: null, intent: null };
      this.ditherTarget = first.intent;
      this.ditherRethinkAt = frameIndex + intRange(this.rng, this.cfg.botDitherRethinkMin, this.cfg.botDitherRethinkMax);
      return first;
    }
    if (frameIndex >= this.ditherRethinkAt) {
      const others = reachable.filter((c) => c.doorwayId !== this.ditherTarget);
      if (others.length > 0) {
        const pick = others[Math.floor(this.rng() * others.length)]!;
        const sigma = this.cfg.botJitterSigmaMin + this.rng() * (this.cfg.botJitterSigmaMax - this.cfg.botJitterSigmaMin);
        this.ditherTarget = pick.doorwayId;
        this.ditherRethinkAt = frameIndex + intRange(this.rng, this.cfg.botDitherRethinkMin, this.cfg.botDitherRethinkMax);
        return { clickX: clamp(pick.screenX + gaussian(this.rng) * sigma, -1, 1), intent: pick.doorwayId };
      }
    }
    return { clickX: null, intent: this.ditherTarget };
  }

  /** Prober: informative positions — midpoints, extremes, 0, wall clicks. */
  private probePosition(reachable: BotState['candidates']): number {
    const xs = reachable.map((c) => c.screenX).sort((a, b) => a - b);
    const options: number[] = [-1, 1, 0];
    for (let i = 1; i < xs.length; i++) {
      options.push((xs[i - 1]! + xs[i]!) / 2); // plateau boundaries
    }
    // A "wall" position far from any candidate: the extreme with the largest
    // distance to the nearest candidate.
    if (xs.length > 0) {
      const nearLeft = xs[0]!;
      const nearRight = xs[xs.length - 1]!;
      const dNeg = Math.abs(-1 - nearLeft);
      const dPos = Math.abs(1 - nearRight);
      options.push(dNeg >= dPos ? -1 : 1);
    }
    return options[Math.floor(this.rng() * options.length)]!;
  }

  /** Spammer: bursts of 2-8 clicks 1-2 frames apart, then 20-80 frames silence. */
  private spammer(frameIndex: number): number | null {
    if (this.spammerBurstRemaining > 0) {
      if (frameIndex >= this.spammerNextAt) {
        this.spammerBurstRemaining -= 1;
        this.spammerNextAt = frameIndex + intRange(this.rng, 1, 2);
        return this.rng() * 2 - 1;
      }
      return null;
    }
    if (frameIndex >= this.spammerNextAt) {
      this.spammerBurstRemaining = intRange(this.rng, this.cfg.botSpammerBurstMin, this.cfg.botSpammerBurstMax);
      this.spammerNextAt = frameIndex; // first click now
      this.spammerBurstRemaining -= 1;
      return this.rng() * 2 - 1;
    }
    if (this.spammerNextAt === 0) {
      this.spammerNextAt = frameIndex + intRange(this.rng, this.cfg.botSpammerSilenceMin, this.cfg.botSpammerSilenceMax);
    }
    return null;
  }
}

/** Build a persona-framework bot for a given mixture profile. */
export function makeBot(seed: number, profile: ProfileName, cfg: EngineConfig): Bot {
  const bot = new ClickBot(seed, profile, cfg);
  return (frameIndex, state) => bot.decide(frameIndex, state);
}

/**
 * LEGACY placeholder bot (kept for the headless unit tests): the old uniform
 * click schedule, exposed through the original `(frameIndex) => number | null`
 * shape.
 */
export function makePlaceholderBot(seed: number): (frameIndex: number) => number | null {
  const bot = makeBot(seed, 'uniformOnly', CONFIG);
  const emptyState: BotState = {
    frameIndex: 0,
    activeEntityId: '',
    player: { x: 0, z: 0, heading: 0 },
    candidates: [],
    controlSignalX: 0,
  };
  return (frameIndex: number) => bot(frameIndex, { ...emptyState, frameIndex }).clickX;
}
