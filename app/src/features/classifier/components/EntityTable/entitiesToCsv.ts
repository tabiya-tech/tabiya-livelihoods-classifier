/**
 * Serialize a Classifier result set to a CSV string suitable for a one-click
 * download. One row per (entity, match) pair so downstream tools can join
 * each ESCO hit back to the source span; empty match rows still surface so
 * the caller sees that the entity was extracted but unlinked.
 *
 * Columns are stable — anything that consumes the file (spreadsheets,
 * Looker, scripts) can rely on the order.
 */

import type { ClassifiedEntity, ClassifyMatch } from "@/lib/api";

export const CSV_COLUMNS = [
  "entity_type",
  "surface_form",
  "span_start",
  "span_end",
  "match_rank",
  "match_label",
  "match_score",
  "match_uri",
] as const;

function escapeCell(value: unknown): string {
  if (value == null) return "";
  const stringified = String(value);
  if (
    stringified.includes(",") ||
    stringified.includes("\"") ||
    stringified.includes("\n") ||
    stringified.includes("\r")
  ) {
    return `"${stringified.replace(/"/g, '""')}"`;
  }
  return stringified;
}

function rowForMatch(
  entity: ClassifiedEntity,
  match: ClassifyMatch | null,
  matchRank: number,
): string {
  const cells = [
    entity.entity_type,
    entity.surface_form,
    entity.span.start,
    entity.span.end,
    match ? matchRank : "",
    match?.entity.preferred_label ?? "",
    match ? match.similarity_score.toFixed(4) : "",
    match?.entity.origin_uri ?? "",
  ];
  return cells.map(escapeCell).join(",");
}

export function entitiesToCsv(entities: ClassifiedEntity[]): string {
  const header = CSV_COLUMNS.join(",");
  const rows: string[] = [header];
  for (const entity of entities) {
    if (entity.matches.length === 0) {
      rows.push(rowForMatch(entity, null, 0));
      continue;
    }
    entity.matches.forEach((match, index) => {
      rows.push(rowForMatch(entity, match, index + 1));
    });
  }
  return rows.join("\n");
}
