import { mergeClassNames } from "@/lib/mergeClassNames";

const uniqueId = "a3f7b291-6c14-4e82-9d05-2f8e1a4c7b53";

export const DATA_TEST_ID = {
  CONTAINER: `stat-card-container-${uniqueId}`,
  LABEL: `stat-card-label-${uniqueId}`,
  VALUE: `stat-card-value-${uniqueId}`,
  DELTA: `stat-card-delta-${uniqueId}`,
};

export interface StatCardProps {
  label: string;
  value: React.ReactNode;
  delta?: string;
  /** Positive delta turns lime; negative turns red. Omit for neutral. */
  deltaPositive?: boolean;
  className?: string;
  "data-testid"?: string;
}

export function StatCard({
  label,
  value,
  delta,
  deltaPositive,
  className,
  "data-testid": dataTestId,
}: StatCardProps) {
  return (
    <div
      data-testid={dataTestId ?? DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        "rounded-md border border-line bg-paper p-5",
        className,
      )}
    >
      <p
        data-testid={DATA_TEST_ID.LABEL}
        className="font-mono text-[11px] font-medium uppercase tracking-wider text-muted"
      >
        {label}
      </p>
      <p
        data-testid={DATA_TEST_ID.VALUE}
        className="mt-1.5 font-mono text-3xl font-semibold text-navy"
      >
        {value}
      </p>
      {delta && (
        <p
          data-testid={DATA_TEST_ID.DELTA}
          className={mergeClassNames(
            "mt-1 font-mono text-[11px]",
            deltaPositive === true && "text-teal",
            deltaPositive === false && "text-error",
            deltaPositive === undefined && "text-muted",
          )}
        >
          {delta}
        </p>
      )}
    </div>
  );
}
