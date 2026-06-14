import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { Table, DATA_TEST_ID } from "./Table";

describe("Table", () => {
  it("renders headers and rows in the right semantic structure", () => {
    // GIVEN expected header labels and body cell values
    const givenHeaderLabels = ["When", "Title"];
    const givenBodyCellValues = ["2 min ago", "Senior Data Engineer"];

    // WHEN we render a Table composed of those headers and a single body row
    render(
      <Table>
        <Table.Head>
          <Table.Row>
            {givenHeaderLabels.map((label) => (
              <Table.HeaderCell key={label}>{label}</Table.HeaderCell>
            ))}
          </Table.Row>
        </Table.Head>
        <Table.Body>
          <Table.Row>
            {givenBodyCellValues.map((value) => (
              <Table.Cell key={value}>{value}</Table.Cell>
            ))}
          </Table.Row>
        </Table.Body>
      </Table>,
    );

    // THEN every compound piece is in the document
    expect(screen.getByTestId(DATA_TEST_ID.TABLE)).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.HEAD)).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.BODY)).toBeInTheDocument();

    // AND the right number of header cells and body cells render
    expect(screen.getAllByTestId(DATA_TEST_ID.HEADER_CELL)).toHaveLength(
      givenHeaderLabels.length,
    );
    expect(screen.getAllByTestId(DATA_TEST_ID.CELL)).toHaveLength(
      givenBodyCellValues.length,
    );
  });
});
