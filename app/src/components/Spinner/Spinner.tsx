import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "8b1f9c2e-4a73-4d51-b7f2-1c0e8d4a9f31";

export const DATA_TEST_ID = {
  CONTAINER: `spinner-container-${uniqueId}`,
};

export interface SpinnerProps {
  /** Pixel size (width and height). Defaults to 14. */
  size?: number;
  className?: string;
  "aria-label"?: string;
}

export function Spinner({
  size = 14,
  className,
  "aria-label": ariaLabel = "Loading",
}: SpinnerProps) {
  return (
    <span
      role="status"
      aria-label={ariaLabel}
      data-testid={DATA_TEST_ID.CONTAINER}
      style={{ width: size, height: size, borderWidth: Math.max(2, size / 7) }}
      className={mergeClassNames(
        "inline-block animate-spin rounded-full border-current border-t-transparent",
        className,
      )}
    />
  );
}
