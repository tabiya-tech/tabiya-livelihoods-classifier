import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";

// Mock the v2 API directly. The repository tests already exercise the
// fetcher; this test focuses on the page composition.
const apiMocks = vi.hoisted(() => {
  let store: Array<{
    key_id: string;
    user_id: string;
    label: string;
    created_at: number;
    last_used_at: number | null;
    revoked: boolean;
  }> = [];
  let nextId = 100;
  return {
    store,
    listApiKeys: vi.fn(async () => ({ keys: store })),
    createApiKey: vi.fn(async (label: string) => {
      const meta = {
        key_id: `key-${nextId++}`,
        user_id: "local-user",
        label,
        created_at: 1_700_000_000,
        last_used_at: null,
        revoked: false,
      };
      store.push(meta);
      return { key: "AIzaSyTEST-PLAINTEXT", meta };
    }),
    deleteApiKey: vi.fn(async (keyId: string) => {
      store = store.filter((row) => row.key_id !== keyId);
      // mutate via reassign — but the closure above holds the old reference,
      // so we have to mutate in place too.
    }),
    seed: (rows: typeof store) => {
      store.length = 0;
      store.push(...rows);
    },
    reset: () => {
      store.length = 0;
      nextId = 100;
    },
  };
});

// Important: hoist a mutating delete that updates the same array reference
// list() reads, so the post-DELETE refetch reflects the change.
apiMocks.deleteApiKey.mockImplementation(async (keyId: string) => {
  const index = apiMocks.store.findIndex((row) => row.key_id === keyId);
  if (index >= 0) apiMocks.store.splice(index, 1);
});

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    listApiKeys: apiMocks.listApiKeys,
    createApiKey: apiMocks.createApiKey,
    deleteApiKey: apiMocks.deleteApiKey,
  };
});

import { ToastProvider } from "@/components";
import { ApiKeysPage, DATA_TEST_ID } from "./ApiKeysPage";
import {
  DATA_TEST_ID as CREATE_FORM_DATA_TEST_ID,
} from "../components/CreateKeyForm/CreateKeyForm";
import {
  DATA_TEST_ID as KEY_TABLE_DATA_TEST_ID,
} from "../components/KeyTable/KeyTable";
import {
  DATA_TEST_ID as REVEAL_BANNER_DATA_TEST_ID,
} from "../components/RevealOnceBanner/RevealOnceBanner";
import {
  DATA_TEST_ID as REVOKE_MODAL_DATA_TEST_ID,
} from "../components/RevokeConfirmModal/RevokeConfirmModal";

function renderApiKeysPage() {
  return render(
    <ToastProvider>
      <ApiKeysPage />
    </ToastProvider>,
  );
}

describe("ApiKeysPage", () => {
  beforeEach(() => {
    apiMocks.reset();
    apiMocks.listApiKeys.mockClear();
    apiMocks.createApiKey.mockClear();
    apiMocks.deleteApiKey.mockClear();
  });

  it("renders the page header and resolves the loading skeleton", async () => {
    // GIVEN the expected title from i18n
    const expectedTitle = i18n.t("keys.title");

    // WHEN we render
    renderApiKeysPage();

    // THEN the title is present and loading clears once the list resolves
    expect(screen.getByTestId(DATA_TEST_ID.TITLE)).toHaveTextContent(
      expectedTitle,
    );
    await waitFor(() =>
      expect(screen.queryByTestId(DATA_TEST_ID.LOADING)).not.toBeInTheDocument(),
    );
  });

  it("shows the empty state when no keys are returned", async () => {
    // GIVEN an empty list response
    // WHEN we render
    renderApiKeysPage();

    // THEN the empty-state element appears once loading completes
    await waitFor(() =>
      expect(
        screen.getByTestId(DATA_TEST_ID.EMPTY_STATE),
      ).toBeInTheDocument(),
    );
  });

  it("creates a new key, reveals it once, and shows it in the table", async () => {
    // GIVEN the expected reveal-banner title
    const expectedBannerTitle = i18n.t("keys.revealBanner.title");
    const givenLabel = "smoke";

    // WHEN we type a label and submit
    renderApiKeysPage();
    await waitFor(() => expect(apiMocks.listApiKeys).toHaveBeenCalled());
    await userEvent.type(
      screen.getByTestId(CREATE_FORM_DATA_TEST_ID.LABEL_INPUT),
      givenLabel,
    );
    await userEvent.click(
      screen.getByTestId(CREATE_FORM_DATA_TEST_ID.SUBMIT_BUTTON),
    );

    // THEN the banner appears with the plaintext key
    const renderedBanner = await screen.findByTestId(
      REVEAL_BANNER_DATA_TEST_ID.CONTAINER,
    );
    expect(renderedBanner).toHaveTextContent(expectedBannerTitle);
    expect(
      screen.getByTestId(REVEAL_BANNER_DATA_TEST_ID.KEY_VALUE),
    ).toHaveTextContent("AIzaSyTEST-PLAINTEXT");

    // AND the new key shows up in the table on refetch
    await waitFor(() => {
      const renderedRows = screen.getAllByTestId(KEY_TABLE_DATA_TEST_ID.ROW);
      expect(renderedRows).toHaveLength(1);
      expect(
        screen.getByTestId(KEY_TABLE_DATA_TEST_ID.LABEL_CELL),
      ).toHaveTextContent(givenLabel);
    });
  });

  it("opens the confirm modal on Revoke and removes the row after confirming", async () => {
    // GIVEN one existing key
    apiMocks.seed([
      {
        key_id: "key-001",
        user_id: "local-user",
        label: "existing",
        created_at: 1_700_000_000,
        last_used_at: null,
        revoked: false,
      },
    ]);

    // WHEN we render, click Revoke, then confirm
    renderApiKeysPage();
    await screen.findByTestId(KEY_TABLE_DATA_TEST_ID.ROW);
    await userEvent.click(
      screen.getByTestId(KEY_TABLE_DATA_TEST_ID.REVOKE_BUTTON),
    );
    expect(
      screen.getByTestId(REVOKE_MODAL_DATA_TEST_ID.CONFIRM_BUTTON),
    ).toBeInTheDocument();
    await userEvent.click(
      screen.getByTestId(REVOKE_MODAL_DATA_TEST_ID.CONFIRM_BUTTON),
    );

    // THEN deleteApiKey was called and the row is gone
    expect(apiMocks.deleteApiKey).toHaveBeenCalledWith("key-001");
    await waitFor(() =>
      expect(
        screen.queryByTestId(KEY_TABLE_DATA_TEST_ID.ROW),
      ).not.toBeInTheDocument(),
    );
  });
});
