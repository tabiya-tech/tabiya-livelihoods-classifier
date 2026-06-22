/**
 * "Are you sure?" before revoking an API key. The page tracks which key is
 * the target via local state; this component is pure presentation.
 */

import { useTranslation } from "react-i18next";
import { Button, Modal } from "@/components";

const uniqueId = "4e7d2b8c-9a1f-4c5d-8e3b-7f2a6d9b1c4e";

export const DATA_TEST_ID = {
  CONFIRM_BUTTON: `revoke-confirm-modal-confirm-${uniqueId}`,
  CANCEL_BUTTON: `revoke-confirm-modal-cancel-${uniqueId}`,
};

export interface RevokeConfirmModalProps {
  /** True when the modal should be visible. */
  open: boolean;
  /** Label of the key being revoked — interpolated into the description. */
  keyLabel: string;
  /** True while the revoke request is in flight. */
  isSubmitting?: boolean;
  /** Fires when the user confirms the revoke. */
  onConfirm: () => void;
  /** Fires when the user cancels (close button or Cancel button). */
  onCancel: () => void;
}

export function RevokeConfirmModal({
  open,
  keyLabel,
  isSubmitting = false,
  onConfirm,
  onCancel,
}: RevokeConfirmModalProps) {
  const { t } = useTranslation();

  return (
    <Modal
      open={open}
      onClose={onCancel}
      title={t("keys.revokeModal.title")}
      description={t("keys.revokeModal.description", { label: keyLabel })}
      footer={
        <>
          <Button
            variant="ghost"
            onClick={onCancel}
            disabled={isSubmitting}
            data-testid={DATA_TEST_ID.CANCEL_BUTTON}
          >
            {t("common.buttons.cancel")}
          </Button>
          <Button
            variant="danger"
            onClick={onConfirm}
            loading={isSubmitting}
            disabled={isSubmitting}
            data-testid={DATA_TEST_ID.CONFIRM_BUTTON}
          >
            {t("keys.revokeModal.confirmLabel")}
          </Button>
        </>
      }
    />
  );
}
