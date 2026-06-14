import { forwardRef, type HTMLAttributes } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "c2f2fc50-0164-42f5-a4d6-6c2d6f16450f";

export const DATA_TEST_ID = {
  CONTAINER: `card-container-${uniqueId}`,
  HEAD_CONTAINER: `card-head-container-${uniqueId}`,
  HEAD_TITLE: `card-head-title-${uniqueId}`,
  HEAD_ACTION: `card-head-action-${uniqueId}`,
};

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Remove the default padding — use when the card hosts its own layout. */
  flush?: boolean;
  /** Use the elevated shadow style instead of flat borders. */
  elevated?: boolean;
}

export const Card = forwardRef<HTMLDivElement, CardProps>(function Card(
  { flush, elevated, className, children, ...rest },
  ref,
) {
  return (
    <div
      ref={ref}
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        "rounded-md border border-line bg-paper",
        !flush && "p-5",
        elevated && "shadow-card-2 border-transparent",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
});

export interface CardHeadProps
  extends Omit<HTMLAttributes<HTMLDivElement>, "title"> {
  title?: React.ReactNode;
  action?: React.ReactNode;
}

export function CardHead({
  title,
  action,
  className,
  children,
  ...rest
}: CardHeadProps) {
  return (
    <div
      data-testid={DATA_TEST_ID.HEAD_CONTAINER}
      className={mergeClassNames(
        "mb-3.5 flex items-center justify-between",
        className,
      )}
      {...rest}
    >
      {title && (
        <h3
          data-testid={DATA_TEST_ID.HEAD_TITLE}
          className="m-0 font-mono text-[13px] font-medium text-navy"
        >
          {title}
        </h3>
      )}
      {children}
      {action && (
        <span data-testid={DATA_TEST_ID.HEAD_ACTION}>{action}</span>
      )}
    </div>
  );
}
