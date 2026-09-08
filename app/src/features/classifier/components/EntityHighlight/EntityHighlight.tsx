/**
 * Renders a source text with inline highlighted spans for each entity.
 *
 * Splitting/overlap logic lives in {@link splitSpans} — this component is
 * the React shell that turns segments into DOM and wires up click +
 * selected/dimmed visual states.
 *
 * Styling: every entity span carries `class="ent"` + `data-type="<entity_type>"`
 * which `.ent[data-type=…]` rules in `src/index.css` colorize.
 */

import type { ClassifiedEntity } from "@/lib/api";
import { mergeClassNames } from "@/lib/mergeClassNames";
import { splitSpans, type EntitySegment } from "./splitSpans";

const uniqueId = "3f7a1d8c-5e2b-4f9d-8a1c-6b3e7d2a4f5c";

export const DATA_TEST_ID = {
  CONTAINER: `entity-highlight-container-${uniqueId}`,
  TEXT_SEGMENT: `entity-highlight-text-segment-${uniqueId}`,
  ENTITY_SEGMENT: `entity-highlight-entity-segment-${uniqueId}`,
};

export interface EntityHighlightProps {
  /** Source text the entities reference. */
  text: string;
  /** Entities to highlight inline. Spans must reference offsets in `text`. */
  entities: ClassifiedEntity[];
  /**
   * The entity index (matches `entityIndex` on segments) that is currently
   * selected — gets a focus ring. Pass null/undefined to deselect.
   */
  selectedEntityIndex?: number | null;
  /**
   * Entity indices that should appear dimmed (e.g. filtered-out type chips).
   * Dimmed entities still render but at reduced opacity.
   */
  dimmedEntityIndices?: ReadonlySet<number>;
  /** Fires when the user clicks any entity span. */
  onEntityClick?: (entity: ClassifiedEntity, entityIndex: number) => void;
  className?: string;
}

export function EntityHighlight({
  text,
  entities,
  selectedEntityIndex = null,
  dimmedEntityIndices,
  onEntityClick,
  className,
}: EntityHighlightProps) {
  const segments = splitSpans(text, entities);

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        "whitespace-pre-wrap font-mono text-[13px] leading-7 text-navy",
        className,
      )}
    >
      {segments.map((segment, segmentIndex) => {
        if (segment.kind === "text") {
          return (
            <span
              key={`t-${segmentIndex}-${segment.start}`}
              data-testid={DATA_TEST_ID.TEXT_SEGMENT}
            >
              {segment.text}
            </span>
          );
        }
        return (
          <EntityHighlightSpan
            key={`e-${segmentIndex}-${segment.start}`}
            segment={segment}
            isSelected={segment.entityIndex === selectedEntityIndex}
            isDimmed={dimmedEntityIndices?.has(segment.entityIndex) ?? false}
            onClick={onEntityClick}
          />
        );
      })}
    </div>
  );
}

interface EntityHighlightSpanProps {
  segment: EntitySegment;
  isSelected: boolean;
  isDimmed: boolean;
  onClick?: (entity: ClassifiedEntity, entityIndex: number) => void;
}

function EntityHighlightSpan({
  segment,
  isSelected,
  isDimmed,
  onClick,
}: EntityHighlightSpanProps) {
  return (
    <span
      role="button"
      tabIndex={0}
      data-testid={DATA_TEST_ID.ENTITY_SEGMENT}
      data-entity-index={segment.entityIndex}
      data-type={segment.entity.entity_type}
      className={mergeClassNames(
        "ent",
        isSelected && "selected",
        isDimmed && "dimmed",
      )}
      onClick={(event) => {
        event.stopPropagation();
        onClick?.(segment.entity, segment.entityIndex);
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onClick?.(segment.entity, segment.entityIndex);
        }
      }}
    >
      {segment.text}
    </span>
  );
}
