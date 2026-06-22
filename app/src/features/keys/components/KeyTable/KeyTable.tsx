/**
 * Tabular listing of the caller's active API keys. Each row exposes a
 * Revoke action — the page wraps the click in the confirm-modal flow.
 *
 * Pure presentation; the snapshot lives in useApiKeys.
 */

import { useTranslation } from "react-i18next";
import { Button, Spinner, Table } from "@/components";
import type { ApiKeyMetadata } from "@/lib/api";
import { mergeClassNames } from "@/lib/mergeClassNames";

const uniqueId = "8a2f5d1c-7b3e-4f9a-8d6c-2e1b4a7d5c3f";

export const DATA_TEST_ID = {
  CONTAINER: `key-table-container-${uniqueId}`,
  ROW: `key-table-row-${uniqueId}`,
  LABEL_CELL: `key-table-label-cell-${uniqueId}`,
  CREATED_CELL: `key-table-created-cell-${uniqueId}`,
  LAST_USED_CELL: `key-table-last-used-cell-${uniqueId}`,
  REVOKE_BUTTON: `key-table-revoke-button-${uniqueId}`,
};

export interface KeyTableProps {
  keys: ApiKeyMetadata[];
  /** Fires when the user clicks Revoke. The page opens the confirm modal. */
  onRevoke: (key: ApiKeyMetadata) => void;
  /** key_id currently being revoked — disables that row's button + shows spinner. */
  pendingKeyId?: string | null;
  className?: string;
}

function formatEpoch(epochSeconds: number, locale: string): string {
  return new Date(epochSeconds * 1000).toLocaleDateString(locale, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function KeyTable({
  keys,
  onRevoke,
  pendingKeyId = null,
  className,
}: KeyTableProps) {
  const { t, i18n } = useTranslation();
  const locale = i18n.language;

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        "overflow-hidden rounded-md border border-line bg-paper",
        className,
      )}
    >
      <Table>
        <Table.Head>
          <Table.Row>
            <Table.HeaderCell>{t("keys.table.headerLabel")}</Table.HeaderCell>
            <Table.HeaderCell>
              {t("keys.table.headerCreated")}
            </Table.HeaderCell>
            <Table.HeaderCell>
              {t("keys.table.headerLastUsed")}
            </Table.HeaderCell>
            <Table.HeaderCell className="w-[1%] text-right">
              {t("keys.table.headerActions")}
            </Table.HeaderCell>
          </Table.Row>
        </Table.Head>
        <Table.Body>
          {keys.map((key) => {
            const isPending = key.key_id === pendingKeyId;
            return (
              <Table.Row
                key={key.key_id}
                data-testid={DATA_TEST_ID.ROW}
                data-key-id={key.key_id}
              >
                <Table.Cell data-testid={DATA_TEST_ID.LABEL_CELL}>
                  <span className="font-mono text-[13px] text-navy">
                    {key.label}
                  </span>
                </Table.Cell>
                <Table.Cell data-testid={DATA_TEST_ID.CREATED_CELL}>
                  <span className="text-xs text-muted">
                    {formatEpoch(key.created_at, locale)}
                  </span>
                </Table.Cell>
                <Table.Cell data-testid={DATA_TEST_ID.LAST_USED_CELL}>
                  <span className="text-xs text-muted">
                    {key.last_used_at == null
                      ? t("keys.table.lastUsedNever")
                      : formatEpoch(key.last_used_at, locale)}
                  </span>
                </Table.Cell>
                <Table.Cell className="text-right">
                  <Button
                    size="sm"
                    variant="danger"
                    disabled={isPending}
                    onClick={() => onRevoke(key)}
                    data-testid={DATA_TEST_ID.REVOKE_BUTTON}
                    leading={isPending ? <Spinner size={12} /> : undefined}
                  >
                    {t("keys.table.revokeButton")}
                  </Button>
                </Table.Cell>
              </Table.Row>
            );
          })}
        </Table.Body>
      </Table>
    </div>
  );
}
