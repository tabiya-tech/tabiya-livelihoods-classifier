import type { HTMLAttributes } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";
import type { EntityType } from "../../theme/theme";

const uniqueId = "4f7b1d92-8c63-4ea5-9027-3b8f5e6d1a04";

export const DATA_TEST_ID = {
  CONTAINER: `score-bar-container-${uniqueId}`,
  FILL: `score-bar-fill-${uniqueId}`,
};

export interface ScoreBarProps extends HTMLAttributes<HTMLDivElement> {
  /** Score between 0 and 1. Values outside the range are clamped. */
  score: number;
  /** Color the fill via the entity palette. Defaults to navy. */
  entityType?: EntityType;
  /** Bar width in pixels. Defaults to 56. */
  width?: number;
}

const fillByType: Record<EntityType, string> = {
  occupation: "bg-entity-occupation-fg",
  skill: "bg-entity-skill-fg",
  qualification: "bg-entity-qualification-fg",
  experience: "bg-entity-experience-fg",
  domain: "bg-entity-domain-fg",
};

export function ScoreBar({
  score,
  entityType,
  width = 56,
  className,
  ...rest
}: ScoreBarProps) {
  const clamped = Math.max(0, Math.min(1, score));
  const pct = `${(clamped * 100).toFixed(1)}%`;
  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={1}
      aria-valuenow={clamped}
      data-testid={DATA_TEST_ID.CONTAINER}
      style={{ width }}
      className={mergeClassNames("h-1 overflow-hidden rounded-sm bg-line", className)}
      {...rest}
    >
      <div
        data-testid={DATA_TEST_ID.FILL}
        style={{ width: pct }}
        className={mergeClassNames(
          "h-full",
          entityType ? fillByType[entityType] : "bg-navy",
        )}
      />
    </div>
  );
}
