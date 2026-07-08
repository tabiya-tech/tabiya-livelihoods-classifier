const uniqueId = "a2e3f7c1-bb4d-4a9e-8d62-3e1f8c9a0b55";

export const DATA_TEST_ID = {
  BADGE: `validation-badge-${uniqueId}`,
  DOT: `validation-badge-dot-${uniqueId}`,
  COUNT: `validation-badge-count-${uniqueId}`,
};

const SEVERITY_STYLES: Record<
  ValidationBadgeSeverity,
  { dotColor: string; textColor: string; backgroundColor: string }
> = {
  info: {
    dotColor: "#26887d",
    textColor: "#26887d",
    backgroundColor: "#d4ebe6",
  },
  warning: {
    dotColor: "#b8860b",
    textColor: "#b8860b",
    backgroundColor: "#f3ecc7",
  },
  error: {
    dotColor: "#c0392b",
    textColor: "#c0392b",
    backgroundColor: "#f4d8dc",
  },
};

export type ValidationBadgeSeverity = "info" | "warning" | "error";

export interface ValidationBadgeProps {
  severity: ValidationBadgeSeverity;
  count: number;
  title?: string;
}

export function ValidationBadge({ severity, count, title }: ValidationBadgeProps) {
  const styles = SEVERITY_STYLES[severity];

  return (
    <span
      data-testid={DATA_TEST_ID.BADGE}
      title={title}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "4px",
        padding: "2px 6px",
        borderRadius: "10px",
        backgroundColor: styles.backgroundColor,
        fontSize: "11px",
        fontWeight: 500,
        color: styles.textColor,
        lineHeight: 1.4,
      }}
    >
      <span
        data-testid={DATA_TEST_ID.DOT}
        style={{
          width: "6px",
          height: "6px",
          borderRadius: "50%",
          backgroundColor: styles.dotColor,
          flexShrink: 0,
        }}
      />
      <span data-testid={DATA_TEST_ID.COUNT}>{count}</span>
    </span>
  );
}
