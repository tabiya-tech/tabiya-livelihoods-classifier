/**
 * A small color dot signaling an entity's type. Uses the same per-type
 * palette as `.ent` via `data-type` on the dot, so the colors stay in
 * lock-step with inline highlights.
 */

import type { ClassifyEntityType } from "@/lib/api";
import { mergeClassNames } from "@/lib/mergeClassNames";

const uniqueId = "2a8d4f7c-1b5e-4c9d-8e2f-7a3b6d1c4e8f";

export const DATA_TEST_ID = {
  DOT: `entity-swatch-dot-${uniqueId}`,
};

export interface EntitySwatchProps {
  entityType: ClassifyEntityType;
  size?: number;
  className?: string;
}

export function EntitySwatch({
  entityType,
  size = 8,
  className,
}: EntitySwatchProps) {
  return (
    <span
      aria-hidden
      data-testid={DATA_TEST_ID.DOT}
      data-type={entityType}
      className={mergeClassNames("ent inline-block rounded-full", className)}
      style={{ width: size, height: size, padding: 0, margin: 0 }}
    />
  );
}
