/**
 * Placeholder component for stories that need to occupy a slot without
 * dragging in a real feature page. Renders a dashed box with diagonal
 * cross-lines and a centered label, so the surrounding layout is obviously
 * the focus of the story.
 *
 * Lives in _test_utilities/ because it is only ever rendered by stories and
 * tests — production code never imports it.
 */

import { mergeClassNames } from "@/lib/mergeClassNames";

const uniqueId = "f7c8d9e0-1a2b-3c4d-5e6f-7a8b9c0d1e2f";

export const DATA_TEST_ID = {
  CONTAINER: `visual-mock-container-${uniqueId}`,
  LABEL: `visual-mock-label-${uniqueId}`,
};

export interface VisualMockProps {
  /** Text shown centered inside the placeholder. */
  text: string;
  /** Optional max-width cap; defaults to filling the parent. */
  maxWidth?: string | number;
  className?: string;
}

export function VisualMock({ text, maxWidth, className }: VisualMockProps) {
  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      style={{ maxWidth }}
      className={mergeClassNames(
        "relative flex h-full min-h-[200px] flex-1 flex-col items-center justify-center",
        "border border-dashed border-line-strong bg-paper text-muted",
        className,
      )}
    >
      <span
        data-testid={DATA_TEST_ID.LABEL}
        className="h-page break-words text-center"
      >
        {text}
      </span>
      <svg
        aria-hidden
        xmlns="http://www.w3.org/2000/svg"
        width="100%"
        height="100%"
        className="pointer-events-none absolute inset-0"
      >
        <line
          x1="0"
          y1="0"
          x2="100%"
          y2="100%"
          stroke="currentColor"
          strokeWidth="1"
          strokeDasharray="2 2"
          opacity="0.35"
        />
        <line
          x1="0"
          y1="100%"
          x2="100%"
          y2="0"
          stroke="currentColor"
          strokeWidth="1"
          strokeDasharray="2 2"
          opacity="0.35"
        />
      </svg>
    </div>
  );
}
