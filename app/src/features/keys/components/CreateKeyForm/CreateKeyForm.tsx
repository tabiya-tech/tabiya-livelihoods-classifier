/**
 * Single-input form for issuing a new API key. The page wires submit into
 * useCreateApiKey; this component is presentation + minimal client validation.
 */

import { useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";
import { Button, FormField, Input } from "@/components";
import { mergeClassNames } from "@/lib/mergeClassNames";

const uniqueId = "9c4f1b2d-3e8a-4d7b-9c1f-5a8e2d4b7c3f";

export const DATA_TEST_ID = {
  CONTAINER: `create-key-form-container-${uniqueId}`,
  LABEL_INPUT: `create-key-form-label-input-${uniqueId}`,
  SUBMIT_BUTTON: `create-key-form-submit-button-${uniqueId}`,
  MAX_REACHED_HELP: `create-key-form-max-reached-help-${uniqueId}`,
};

export interface CreateKeyFormProps {
  /** Fires with the trimmed label when the user submits a valid value. */
  onSubmit: (label: string) => void;
  /** True while a previous submit is in flight; disables interaction. */
  isSubmitting?: boolean;
  /** True when the user has hit the per-account key limit. */
  maxReached?: boolean;
  /** Limit shown to the user when maxReached is true. */
  maxKeys?: number;
  className?: string;
}

export function CreateKeyForm({
  onSubmit,
  isSubmitting = false,
  maxReached = false,
  maxKeys,
  className,
}: CreateKeyFormProps) {
  const { t } = useTranslation();
  const [label, setLabel] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = label.trim();
    if (!trimmed || isSubmitting || maxReached) return;
    onSubmit(trimmed);
    setLabel("");
  }

  const submitDisabled = isSubmitting || maxReached || label.trim().length === 0;

  return (
    <form
      data-testid={DATA_TEST_ID.CONTAINER}
      onSubmit={handleSubmit}
      className={mergeClassNames(
        "flex w-full flex-col gap-3 sm:flex-row sm:items-end",
        className,
      )}
    >
      <FormField
        label={t("keys.createForm.labelLabel")}
        className="flex-1"
        help={
          maxReached && maxKeys != null
            ? (
              <span data-testid={DATA_TEST_ID.MAX_REACHED_HELP}>
                {t("keys.createForm.maxReachedHelp", { max: maxKeys })}
              </span>
            )
            : undefined
        }
      >
        <Input
          data-testid={DATA_TEST_ID.LABEL_INPUT}
          value={label}
          onChange={(event) => setLabel(event.target.value)}
          placeholder={t("keys.createForm.labelPlaceholder")}
          disabled={isSubmitting || maxReached}
          maxLength={100}
        />
      </FormField>
      <Button
        type="submit"
        variant="primary"
        loading={isSubmitting}
        disabled={submitDisabled}
        data-testid={DATA_TEST_ID.SUBMIT_BUTTON}
      >
        {t("keys.createForm.submit")}
      </Button>
    </form>
  );
}
