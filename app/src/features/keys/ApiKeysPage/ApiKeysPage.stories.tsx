import { useState } from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { MemoryRouter } from "react-router-dom";
import { ToastProvider } from "@/components";
import type { ApiKeyMetadata, CreateApiKeyResponse } from "@/lib/api";
import { fixtureApiKeyMetadata } from "@/mocks/fixtures/apiKeys";
import { ApiKeysOverridesProvider } from "../hooks/apiKeysOverrides";
import type { ApiKeysSnapshot } from "../hooks/useApiKeys";
import type {
  CreateApiKeyState,
  CreateApiKeyStatus,
} from "../hooks/useCreateApiKey";
import type {
  RevokeApiKeyState,
  RevokeApiKeyStatus,
} from "../hooks/useRevokeApiKey";
import { ApiKeysPage } from "./ApiKeysPage";

interface ApiKeysHarnessProps {
  seedKeys: ApiKeyMetadata[];
  initialJustIssued?: CreateApiKeyResponse;
  createStatus?: CreateApiKeyStatus;
  revokeStatus?: RevokeApiKeyStatus;
  onCreate?: (label: string) => Promise<CreateApiKeyResponse>;
  onRevoke?: (keyId: string) => Promise<void>;
  children: React.ReactNode;
}

/**
 * Stateful harness wrapping the page with stub hook implementations so
 * stories are fully interactive without a backend.
 */
function ApiKeysHarness({
  seedKeys,
  initialJustIssued,
  createStatus = "idle",
  revokeStatus = "idle",
  onCreate,
  onRevoke,
  children,
}: ApiKeysHarnessProps) {
  const [keys, setKeys] = useState<ApiKeyMetadata[]>(seedKeys);
  const [justIssued, setJustIssued] = useState<CreateApiKeyResponse | null>(
    initialJustIssued ?? null,
  );
  const [currentCreateStatus, setCreateStatus] =
    useState<CreateApiKeyStatus>(createStatus);
  const [currentRevokeStatus, setRevokeStatus] =
    useState<RevokeApiKeyStatus>(revokeStatus);

  const apiKeysOverride: ApiKeysSnapshot = {
    status: "ready",
    keys,
    error: null,
    refetch: async () => undefined,
  };

  const createApiKeyOverride: CreateApiKeyState = {
    status: currentCreateStatus,
    error: null,
    justIssued,
    submit: async (label: string) => {
      setCreateStatus("submitting");
      const response = onCreate
        ? await onCreate(label)
        : {
            key: "AIzaSyDEMO0000000000000000000000000000000",
            meta: {
              key_id: `story-${Math.floor(label.length * 7)}`,
              user_id: "local-user",
              label,
              created_at: 1_700_000_000,
              last_used_at: null,
              revoked: false,
            },
          };
      setKeys((previous) => [...previous, response.meta]);
      setJustIssued(response);
      setCreateStatus("success");
      return response;
    },
    clearJustIssuedKey: () => {
      setJustIssued(null);
      setCreateStatus("idle");
    },
  };

  const revokeApiKeyOverride: RevokeApiKeyState = {
    status: currentRevokeStatus,
    error: null,
    pendingKeyId: null,
    revoke: async (keyId: string) => {
      setRevokeStatus("submitting");
      if (onRevoke) await onRevoke(keyId);
      setKeys((previous) => previous.filter((row) => row.key_id !== keyId));
      setRevokeStatus("idle");
    },
  };

  return (
    <ApiKeysOverridesProvider
      apiKeys={apiKeysOverride}
      createApiKey={createApiKeyOverride}
      revokeApiKey={revokeApiKeyOverride}
    >
      {children}
    </ApiKeysOverridesProvider>
  );
}

const meta: Meta<typeof ApiKeysPage> = {
  title: "Features/Keys/ApiKeysPage",
  component: ApiKeysPage,
  parameters: { layout: "fullscreen" },
  decorators: [
    function StoryWithProviders(StoryComponent) {
      return (
        <MemoryRouter initialEntries={["/keys"]}>
          <ToastProvider>
            <StoryComponent />
          </ToastProvider>
        </MemoryRouter>
      );
    },
  ],
};
export default meta;

type Story = StoryObj<typeof ApiKeysPage>;

export const WithKeys: Story = {
  render: function WithKeysStory() {
    return (
      <ApiKeysHarness seedKeys={fixtureApiKeyMetadata}>
        <ApiKeysPage />
      </ApiKeysHarness>
    );
  },
};

export const Empty: Story = {
  render: function EmptyStory() {
    return (
      <ApiKeysHarness seedKeys={[]}>
        <ApiKeysPage />
      </ApiKeysHarness>
    );
  },
};

/**
 * Five active keys — the per-account limit. The Create button is disabled
 * and a helper appears under the input.
 */
export const MaxReached: Story = {
  render: function MaxReachedStory() {
    const fiveKeys: ApiKeyMetadata[] = Array.from({ length: 5 }, (_unused, index) => ({
      key_id: `max-${index}`,
      user_id: "local-user",
      label: `slot-${index + 1}`,
      created_at: 1_700_000_000 + index * 1000,
      last_used_at: null,
      revoked: false,
    }));
    return (
      <ApiKeysHarness seedKeys={fiveKeys}>
        <ApiKeysPage />
      </ApiKeysHarness>
    );
  },
};

/**
 * Reveal-once banner pinned at the top from a just-issued key.
 */
export const JustCreated: Story = {
  render: function JustCreatedStory() {
    return (
      <ApiKeysHarness
        seedKeys={fixtureApiKeyMetadata}
        initialJustIssued={{
          key: "AIzaSyDEMO0000000000000000000000000000000",
          meta: {
            key_id: "story-new",
            user_id: "local-user",
            label: "freshly-issued",
            created_at: 1_700_999_999,
            last_used_at: null,
            revoked: false,
          },
        }}
        createStatus="success"
      >
        <ApiKeysPage />
      </ApiKeysHarness>
    );
  },
};

/**
 * Backend that rejects every create. Surfaces the error toast.
 */
export const CreateError: Story = {
  render: function CreateErrorStory() {
    const onCreate = fn(async () => {
      throw new Error("Backend unreachable");
    });
    return (
      <ApiKeysHarness
        seedKeys={fixtureApiKeyMetadata}
        onCreate={onCreate as never}
      >
        <ApiKeysPage />
      </ApiKeysHarness>
    );
  },
};
