import type {
  HTMLAttributes,
  TableHTMLAttributes,
  TdHTMLAttributes,
  ThHTMLAttributes,
} from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "e11822ab-7075-4a77-b185-7bc349e3d2c1";

export const DATA_TEST_ID = {
  TABLE: `table-table-${uniqueId}`,
  HEAD: `table-head-${uniqueId}`,
  BODY: `table-body-${uniqueId}`,
  ROW: `table-row-${uniqueId}`,
  HEADER_CELL: `table-header-cell-${uniqueId}`,
  CELL: `table-cell-${uniqueId}`,
};

export type TableProps = TableHTMLAttributes<HTMLTableElement>;

export function Table({ className, ...rest }: TableProps) {
  return (
    <table
      data-testid={DATA_TEST_ID.TABLE}
      className={mergeClassNames("w-full border-collapse text-[13px]", className)}
      {...rest}
    />
  );
}

Table.Head = function Thead(props: HTMLAttributes<HTMLTableSectionElement>) {
  return <thead data-testid={DATA_TEST_ID.HEAD} {...props} />;
};

Table.Body = function Tbody(props: HTMLAttributes<HTMLTableSectionElement>) {
  return <tbody data-testid={DATA_TEST_ID.BODY} {...props} />;
};

export type TableRowProps = HTMLAttributes<HTMLTableRowElement> & {
  /** Hover highlight (e.g. when row is clickable). */
  hover?: boolean;
};

Table.Row = function Tr({ hover, className, ...rest }: TableRowProps) {
  return (
    <tr
      data-testid={DATA_TEST_ID.ROW}
      className={mergeClassNames(
        hover && "hover:bg-[rgba(0,33,71,0.02)] cursor-pointer",
        className,
      )}
      {...rest}
    />
  );
};

Table.HeaderCell = function Th({
  className,
  ...rest
}: ThHTMLAttributes<HTMLTableCellElement>) {
  return (
    <th
      data-testid={DATA_TEST_ID.HEADER_CELL}
      className={mergeClassNames(
        "border-b border-line bg-cream px-3.5 py-2.5 text-left",
        "font-mono text-[11px] font-medium uppercase tracking-wider text-muted",
        className,
      )}
      {...rest}
    />
  );
};

Table.Cell = function Td({
  className,
  ...rest
}: TdHTMLAttributes<HTMLTableCellElement>) {
  return (
    <td
      data-testid={DATA_TEST_ID.CELL}
      className={mergeClassNames(
        "border-b border-line px-3.5 py-3.5 align-middle",
        "[tr:last-child_&]:border-b-0",
        className,
      )}
      {...rest}
    />
  );
};
