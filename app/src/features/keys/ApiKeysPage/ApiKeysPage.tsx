/**
 * Tabiya API Keys page — list / create / revoke against /v2/user/api-keys.
 *
 * Composition:
 *   - {@link CreateKeyForm} above the table.
 *   - {@link RevealOnceBanner} appears once a POST returns the plaintext key.
 *   - {@link KeyTable} renders active keys; clicking Revoke opens the modal.
 *   - {@link RevokeConfirmModal} confirms; on Confirm we hit DELETE then refetch.
 *
 * The page owns:
 *   - which key is the revoke target (local state).
 *   - error toasts for create / revoke / load failures.
 */

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { EmptyState, Spinner, useToast } from "@/components";
import type { ApiKeyMetadata } from "@/lib/api";
import { CreateKeyForm } from "../components/CreateKeyForm/CreateKeyForm";
import { KeyTable } from "../components/KeyTable/KeyTable";
import { RevealOnceBanner } from "../components/RevealOnceBanner/RevealOnceBanner";
import { RevokeConfirmModal } from "../components/RevokeConfirmModal/RevokeConfirmModal";
import { useApiKeys } from "../hooks/useApiKeys";
import { useCreateApiKey } from "../hooks/useCreateApiKey";
import { useRevokeApiKey } from "../hooks/useRevokeApiKey";

const uniqueId = "6a3f9d2c-1e7b-4d8a-9c5f-2b8e4d7a1c3f";

export const DATA_TEST_ID = {
  CONTAINER: `api-keys-page-container-${uniqueId}`,
  EYEBROW: `api-keys-page-eyebrow-${uniqueId}`,
  TITLE: `api-keys-page-title-${uniqueId}`,
  INTRO: `api-keys-page-intro-${uniqueId}`,
  SECTION_TITLE: `api-keys-page-section-title-${uniqueId}`,
  SECTION_INTRO: `api-keys-page-section-intro-${uniqueId}`,
  LOADING: `api-keys-page-loading-${uniqueId}`,
  LOAD_ERROR: `api-keys-page-load-error-${uniqueId}`,
  EMPTY_STATE: `api-keys-page-empty-state-${uniqueId}`,
};

/** Per-account upper bound. Mirrors the backend's MAX_API_KEYS_PER_USER. */
const MAX_KEYS_PER_USER = 5;

export function ApiKeysPage() {
  const { t } = useTranslation();
  const toast = useToast();

  const apiKeys = useApiKeys();
  const createKey = useCreateApiKey({
    onSuccess: () => apiKeys.refetch(),
  });
  const revokeKey = useRevokeApiKey({
    onSuccess: () => apiKeys.refetch(),
  });

  const [revokeTarget, setRevokeTarget] = useState<ApiKeyMetadata | null>(null);

  async function handleCreate(label: string) {
    try {
      await createKey.submit(label);
    } catch {
      toast.show({
        message: t("keys.toasts.createError"),
        tone: "error",
      });
    }
  }

  async function handleConfirmRevoke() {
    if (!revokeTarget) return;
    try {
      await revokeKey.revoke(revokeTarget.key_id);
      setRevokeTarget(null);
    } catch {
      toast.show({
        message: t("keys.toasts.revokeError"),
        tone: "error",
      });
    }
  }

  const isLoading = apiKeys.status === "loading";
  const loadError = apiKeys.error;
  const activeKeys = apiKeys.keys;
  const maxReached = activeKeys.length >= MAX_KEYS_PER_USER;

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className="mx-auto flex w-full max-w-[1080px] flex-col gap-8 px-4 py-6 sm:px-8 sm:py-10"
    >
      <header className="flex flex-col gap-2">
        <span data-testid={DATA_TEST_ID.EYEBROW} className="eyebrow">
          {t("keys.eyebrow")}
        </span>
        <h1 data-testid={DATA_TEST_ID.TITLE} className="h-page m-0">
          {t("keys.title")}
        </h1>
        <p
          data-testid={DATA_TEST_ID.INTRO}
          className="m-0 max-w-[680px] text-sm leading-relaxed text-muted"
        >
          {t("keys.intro")}
        </p>
      </header>

      {createKey.justIssued && (
        <RevealOnceBanner
          apiKey={createKey.justIssued.key}
          onDismiss={createKey.clearJustIssuedKey}
        />
      )}

      <section className="flex flex-col gap-4">
        <header className="flex flex-col gap-1">
          <h2
            data-testid={DATA_TEST_ID.SECTION_TITLE}
            className="h-section m-0"
          >
            {t("keys.tabiyaSectionTitle")}
          </h2>
          <p
            data-testid={DATA_TEST_ID.SECTION_INTRO}
            className="m-0 text-xs text-muted"
          >
            {t("keys.tabiyaSectionIntro", {
              header: t("keys.tabiyaSectionHeader"),
              endpoint: t("keys.tabiyaSectionEndpoint"),
            })}
          </p>
        </header>

        <CreateKeyForm
          onSubmit={(label) => {
            void handleCreate(label);
          }}
          isSubmitting={createKey.status === "submitting"}
          maxReached={maxReached}
          maxKeys={MAX_KEYS_PER_USER}
        />

        {isLoading && (
          <div
            data-testid={DATA_TEST_ID.LOADING}
            className="flex items-center gap-2 rounded-md border border-line bg-paper px-4 py-3 font-mono text-xs text-muted"
          >
            <Spinner />
            {t("common.loading")}
          </div>
        )}

        {!isLoading && loadError && (
          <div data-testid={DATA_TEST_ID.LOAD_ERROR}>
            <EmptyState icon="close" title={t("keys.toasts.loadError")} />
          </div>
        )}

        {!isLoading && !loadError && activeKeys.length === 0 && (
          <div data-testid={DATA_TEST_ID.EMPTY_STATE}>
            <EmptyState
              icon="key"
              title={t("keys.empty.title")}
              description={t("keys.empty.description")}
            />
          </div>
        )}

        {!isLoading && !loadError && activeKeys.length > 0 && (
          <KeyTable
            keys={activeKeys}
            onRevoke={setRevokeTarget}
            pendingKeyId={revokeKey.pendingKeyId}
          />
        )}
      </section>

      <RevokeConfirmModal
        open={revokeTarget !== null}
        keyLabel={revokeTarget?.label ?? ""}
        isSubmitting={revokeKey.status === "submitting"}
        onConfirm={() => {
          void handleConfirmRevoke();
        }}
        onCancel={() => setRevokeTarget(null)}
      />
    </div>
  );
}
