/**
 * Confirmation modal that fires when the user tries to navigate away from
 * the Configuration page (or any host page) with an unsaved draft.
 *
 * The hosting page owns the trigger logic (router blocker, beforeunload
 * listener, etc.); this component is the pure presentational confirmation.
 */

import { useTranslation } from "react-i18next";
import { Button, Modal } from "@/components";

const uniqueId = "a5b1d8c2-9e4f-4a7b-bf30-7c1e2d8a5b4f";

export const DATA_TEST_ID = {
  CONFIRM_BUTTON: `unsaved-changes-guard-confirm-${uniqueId}`,
  CANCEL_BUTTON: `unsaved-changes-guard-cancel-${uniqueId}`,
};

export interface UnsavedChangesGuardProps {
  /** True when the modal should be visible. */
  open: boolean;
  /** Fires when the user chooses to discard their changes and proceed. */
  onConfirm: () => void;
  /** Fires when the user dismisses the modal and keeps editing. */
  onCancel: () => void;
}

export function UnsavedChangesGuard({
  open,
  onConfirm,
  onCancel,
}: UnsavedChangesGuardProps) {
  const { t } = useTranslation();

  return (
    <Modal
      open={open}
      onClose={onCancel}
      title={t("configuration.unsavedGuard.title")}
      description={t("configuration.unsavedGuard.description")}
      footer={
        <>
          <Button
            variant="ghost"
            onClick={onCancel}
            data-testid={DATA_TEST_ID.CANCEL_BUTTON}
          >
            {t("common.buttons.keepEditing")}
          </Button>
          <Button
            variant="danger"
            onClick={onConfirm}
            data-testid={DATA_TEST_ID.CONFIRM_BUTTON}
          >
            {t("configuration.unsavedGuard.confirmLabel")}
          </Button>
        </>
      }
    />
  );
}
