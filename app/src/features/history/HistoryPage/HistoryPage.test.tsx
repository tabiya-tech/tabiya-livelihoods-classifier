import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import i18n from "@/i18n/i18n";
import {
  resetClassificationsHandlersStore,
  seedClassificationsHandlersStore,
  fixtureClassificationSummaries,
} from "@/mocks/handlers";
import { HistoryPage, DATA_TEST_ID } from "./HistoryPage";

function renderHistory() {
  return render(
    <MemoryRouter>
      <HistoryPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  resetClassificationsHandlersStore();
});

describe("HistoryPage", () => {
  it("renders a spinner while loading", () => {
    // GIVEN the page is loading (network not yet settled)
    renderHistory();

    // WHEN the component first mounts
    // THEN a spinner is visible
    expect(screen.getByTestId(DATA_TEST_ID.SPINNER)).toBeInTheDocument();
  });

  it("renders the classifications table once data loads", async () => {
    // GIVEN fixture classifications in the MSW store
    // WHEN rendered
    renderHistory();

    // THEN the table appears
    await waitFor(() =>
      expect(screen.getByTestId(DATA_TEST_ID.TABLE)).toBeInTheDocument(),
    );
  });

  it("renders one row per classification", async () => {
    // GIVEN 5 fixture classifications
    const expectedRowCount = fixtureClassificationSummaries.length;
    renderHistory();

    // WHEN data loads
    await waitFor(() =>
      expect(screen.getByTestId(DATA_TEST_ID.TABLE)).toBeInTheDocument(),
    );

    // THEN each classification has a row (Table.Row includes header row, so +1)
    const rows = screen.getAllByRole("row");
    expect(rows.length - 1).toBe(expectedRowCount);
  });

  it("shows empty state when no classifications exist", async () => {
    // GIVEN no classifications in the store
    seedClassificationsHandlersStore({ classifications: [] });

    // WHEN rendered
    renderHistory();

    // THEN empty state message is shown
    const expectedTitle = i18n.t("history.table.empty");
    await waitFor(() =>
      expect(screen.getByText(expectedTitle)).toBeInTheDocument(),
    );
  });

  it("shows Load more button when next_cursor is present", async () => {
    // GIVEN more items than the default limit — seed 25 items so the handler
    // returns a next_cursor with default limit=20
    const manyItems = Array.from({ length: 25 }, (_, index) => ({
      classification_id: `cls-${index}`,
      pipeline_id: "pipeline-default",
      entity_count: index,
      processing_time_ms: 100,
      created_at: "2026-07-10T10:00:00.000Z",
    }));
    seedClassificationsHandlersStore({ classifications: manyItems });

    // WHEN rendered
    renderHistory();

    // THEN Load more button appears
    await waitFor(() =>
      expect(screen.getByTestId(DATA_TEST_ID.LOAD_MORE)).toBeInTheDocument(),
    );
  });

  it("loads more rows when Load more is clicked", async () => {
    // GIVEN 25 items so that limit=20 leaves 5 on the next page
    const manyItems = Array.from({ length: 25 }, (_, index) => ({
      classification_id: `cls-${index}`,
      pipeline_id: "pipeline-default",
      entity_count: index,
      processing_time_ms: 100,
      created_at: "2026-07-10T10:00:00.000Z",
    }));
    seedClassificationsHandlersStore({ classifications: manyItems });
    const user = userEvent.setup();

    renderHistory();

    // WHEN first page loads and user clicks Load more
    await waitFor(() =>
      expect(screen.getByTestId(DATA_TEST_ID.LOAD_MORE)).toBeInTheDocument(),
    );
    await user.click(screen.getByTestId(DATA_TEST_ID.LOAD_MORE));

    // THEN all 25 rows are visible (header + 25 data rows)
    await waitFor(() => {
      const rows = screen.getAllByRole("row");
      expect(rows.length - 1).toBe(25);
    });
  });
});
