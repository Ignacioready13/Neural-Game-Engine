import type { CandidateDoorway } from '../core/types';

/**
 * Doorway target resolution (Section 6.2 / 6.6).
 *
 * Pure function: given this frame's candidate doorways (each already projected
 * to screen X and pre-flagged reachable), the held click X, and the previous
 * target id, return the id of the doorway to steer toward (or null).
 *
 * Algorithm:
 *   - referenceX = clickX ?? 0 (screen-center default before any click).
 *   - Sort candidates by |screenX - referenceX| ascending (stable tie-break).
 *   - If the nearest candidate is reachable, it wins.
 *   - Otherwise search outward in increasing screen distance, BIASED to stay on
 *     the same side of the reference as the nearest candidate before crossing
 *     to the other side; return the first reachable candidate found.
 *   - If none are reachable, return the previous target if it is still a live
 *     candidate, else null (the caller keeps straight-ahead steering).
 */
export function resolveTarget(
  candidates: CandidateDoorway[],
  clickX: number | null,
  previousTargetId: string | null,
): string | null {
  if (candidates.length === 0) return null;

  const referenceX = clickX ?? 0;

  // Stable sort by screen distance from the reference, tie-broken by original
  // index so the result is deterministic for equal distances.
  const indexed = candidates
    .map((c, i) => ({ c, i, d: Math.abs(c.screenX - referenceX) }))
    .sort((a, b) => a.d - b.d || a.i - b.i);

  const nearest = indexed[0]!.c;
  if (nearest.reachable) return nearest.doorwayId;

  // Which side of the reference did the search start on? left = screenX < ref,
  // else right (a candidate exactly at the reference counts as "right").
  const nearestOnLeft = nearest.screenX < referenceX;

  // The remaining candidates are already in ascending distance order, so
  // splitting them preserves the outward search order within each side.
  const sameSide: CandidateDoorway[] = [];
  const otherSide: CandidateDoorway[] = [];
  for (let k = 1; k < indexed.length; k++) {
    const c = indexed[k]!.c;
    if ((c.screenX < referenceX) === nearestOnLeft) sameSide.push(c);
    else otherSide.push(c);
  }

  for (const c of sameSide) if (c.reachable) return c.doorwayId;
  for (const c of otherSide) if (c.reachable) return c.doorwayId;

  // Nothing reachable: keep the previous target when it is still live.
  if (
    previousTargetId !== null &&
    candidates.some((c) => c.doorwayId === previousTargetId)
  ) {
    return previousTargetId;
  }
  return null;
}
