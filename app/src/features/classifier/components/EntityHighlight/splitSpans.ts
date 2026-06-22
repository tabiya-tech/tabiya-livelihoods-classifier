/**
 * Pure span splitter — turns (text, entities) into a sequence of segments
 * that can be rendered as alternating text and highlighted spans.
 *
 * Why a separate function (no React): unit-testable without rendering, and
 * the same logic is reused by exports (Table tab CSV, etc.).
 *
 * Overlap policy: when two spans overlap, the one with higher
 * `similarity_score` of its top match wins. The other is dropped entirely
 * — we never render partial overlaps because the visual result is
 * confusing and the underlying entity loses meaning when sliced apart.
 *
 * Zero-length and out-of-range spans are dropped silently.
 */

import type { ClassifiedEntity } from "@/lib/api";

export interface PlainTextSegment {
  kind: "text";
  text: string;
  /** Character offset in the source string where this segment starts. */
  start: number;
}

export interface EntitySegment {
  kind: "ent";
  text: string;
  start: number;
  end: number;
  entity: ClassifiedEntity;
  /** Index into the *kept* entities array — useful for click handlers. */
  entityIndex: number;
}

export type SourceSegment = PlainTextSegment | EntitySegment;

interface RankedEntity {
  entity: ClassifiedEntity;
  /** Original index in the caller's input array (so click handlers stay stable). */
  originalIndex: number;
}

function topScore(entity: ClassifiedEntity): number {
  // matches are returned in descending score order, so [0] is the best.
  return entity.matches[0]?.similarity_score ?? 0;
}

/**
 * Drop overlapping entities, preferring the higher-scoring one. Returns the
 * survivors sorted by `start` ascending.
 */
function resolveOverlaps(
  ranked: RankedEntity[],
  textLength: number,
): RankedEntity[] {
  const filtered = ranked.filter(
    (item) =>
      item.entity.span.start >= 0 &&
      item.entity.span.end <= textLength &&
      item.entity.span.end > item.entity.span.start,
  );

  // Sort by score desc so the loop picks survivors greedily by quality.
  const byScoreDesc = [...filtered].sort(
    (left, right) => topScore(right.entity) - topScore(left.entity),
  );

  const survivors: RankedEntity[] = [];
  for (const candidate of byScoreDesc) {
    const overlaps = survivors.some((existing) => {
      const a = candidate.entity.span;
      const b = existing.entity.span;
      return a.start < b.end && b.start < a.end;
    });
    if (!overlaps) survivors.push(candidate);
  }

  return survivors.sort(
    (left, right) => left.entity.span.start - right.entity.span.start,
  );
}

export function splitSpans(
  text: string,
  entities: ClassifiedEntity[],
): SourceSegment[] {
  const ranked: RankedEntity[] = entities.map((entity, index) => ({
    entity,
    originalIndex: index,
  }));
  const survivors = resolveOverlaps(ranked, text.length);

  const segments: SourceSegment[] = [];
  let cursor = 0;

  for (const survivor of survivors) {
    const { entity, originalIndex } = survivor;
    if (entity.span.start > cursor) {
      segments.push({
        kind: "text",
        text: text.slice(cursor, entity.span.start),
        start: cursor,
      });
    }
    segments.push({
      kind: "ent",
      text: text.slice(entity.span.start, entity.span.end),
      start: entity.span.start,
      end: entity.span.end,
      entity,
      entityIndex: originalIndex,
    });
    cursor = entity.span.end;
  }

  if (cursor < text.length) {
    segments.push({
      kind: "text",
      text: text.slice(cursor),
      start: cursor,
    });
  }

  return segments;
}