import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";
import type { ApiKeyMetadata } from "@/lib/api";
import { DATA_TEST_ID, KeyTable } from "./KeyTable";

const givenKey: ApiKeyMetadata = {
  key_id: "key-001",
  user_id: "u",
  label: "analyst-laptop",
  created_at: 1_700_000_000,
  last_used_at: 1_700_100_000,
  revoked: false,
};

const givenKeyNeverUsed: ApiKeyMetadata = {
  key_id: "key-002",
  user_id: "u",
  label: "ci-pipeline",
  created_at: 1_700_050_000,
  last_used_at: null,
  revoked: false,
};

describe("KeyTable", () => {
  it("renders one row per key with label + dates", () => {
    // GIVEN two keys
    const givenKeys = [givenKey, givenKeyNeverUsed];

    // WHEN we render
    render(<KeyTable keys={givenKeys} onRevoke={() => {}} />);

    // THEN the table has two rows, one per key
    const renderedRows = screen.getAllByTestId(DATA_TEST_ID.ROW);
    expect(renderedRows).toHaveLength(2);
    expect(renderedRows[0]).toHaveAttribute("data-key-id", givenKey.key_id);
    expect(within(renderedRows[0]).getByTestId(DATA_TEST_ID.LABEL_CELL))
      .toHaveTextContent(givenKey.label);
  });

  it("shows the localized 'Never' marker when last_used_at is null", () => {
    // GIVEN a key that's never been used and the expected label
    const expectedNeverLabel = i18n.t("keys.table.lastUsedNever");

    // WHEN we render
    render(<KeyTable keys={[givenKeyNeverUsed]} onRevoke={() => {}} />);

    // THEN the last-used cell shows the Never marker
    expect(screen.getByTestId(DATA_TEST_ID.LAST_USED_CELL)).toHaveTextContent(
      expectedNeverLabel,
    );
  });

  it("invokes onRevoke with the full metadata when the revoke button is clicked", async () => {
    // GIVEN an onRevoke spy and one row
    const onRevoke = vi.fn();
    render(<KeyTable keys={[givenKey]} onRevoke={onRevoke} />);

    // WHEN the revoke button is clicked
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.REVOKE_BUTTON));

    // THEN onRevoke fires with the metadata
    expect(onRevoke).toHaveBeenCalledWith(givenKey);
  });

  it("disables the revoke button for the row matching pendingKeyId", () => {
    // GIVEN two rows and a pending revoke for the first one
    const givenKeys = [givenKey, givenKeyNeverUsed];

    // WHEN we render with pendingKeyId set to key-001
    render(
      <KeyTable
        keys={givenKeys}
        onRevoke={() => {}}
        pendingKeyId={givenKey.key_id}
      />,
    );

    // THEN the first row's revoke button is disabled and the second is enabled
    const renderedButtons = screen.getAllByTestId(DATA_TEST_ID.REVOKE_BUTTON);
    expect(renderedButtons[0]).toBeDisabled();
    expect(renderedButtons[1]).toBeEnabled();
  });
});
