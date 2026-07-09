/**
 * Fixed-bottom save bar for the pipeline editor.
 *
 * Layout (left-to-right):
 *   [Name input] | [Validation pill] | [Cancel] [Save]
 *
 * The name input is hidden when the pipeline is readonly.
 * The validation pill reflects debounce / issue / clean state.
 * The save button is disabled when saving, invalid, clean, or readonly.
 */

import { useTranslation } from "react-i18next";
import { Button, Input } from "@/components";
import type { PipelineValidationIssue } from "@/lib/api";

const uniqueId = "b7c2d4e6-f8a0-4b1c-9d3e-5f7a8b9c0d1e";

export const DATA_TEST_ID = {
  ROOT: `pipeline-save-bar-root-${uniqueId}`,
  NAME_INPUT: `pipeline-save-bar-name-input-${uniqueId}`,
  VALIDATION_PILL: `pipeline-save-bar-validation-pill-${uniqueId}`,
  CANCEL_BUTTON: `pipeline-save-bar-cancel-button-${uniqueId}`,
  SAVE_BUTTON: `pipeline-save-bar-save-button-${uniqueId}`,
};

export interface PipelineSaveBarProps {
  name: string;
  onNameChange: (nextName: string) => void;
  isReadonly: boolean;
  /** Validation issues from useValidatePipeline. */
  issues: PipelineValidationIssue[];
  /** True while the debounce window is running (status === "checking"). */
  isValidating: boolean;
  /** True when current state differs from last-saved state. */
  isDirty: boolean;
  /** True while the save request is in-flight. */
  isSaving: boolean;
  onSave: () => void;
  onCancel: () => void;
}

function ValidationPill({
  isValidating,
  issues,
}: {
  isValidating: boolean;
  issues: PipelineValidationIssue[];
}) {
  const { t } = useTranslation();

  if (isValidating) {
    return (
      <span
        data-testid={DATA_TEST_ID.VALIDATION_PILL}
        className="inline-flex items-center rounded-full bg-muted/20 px-3 py-1 text-xs font-medium text-muted"
      >
        {t("pipelines.editor.saveBar.validating")}
      </span>
    );
  }

  if (issues.length === 0) {
    return (
      <span
        data-testid={DATA_TEST_ID.VALIDATION_PILL}
        className="inline-flex items-center rounded-full bg-green-100 px-3 py-1 text-xs font-medium text-green-700"
      >
        {t("pipelines.editor.saveBar.valid")}
      </span>
    );
  }

  return (
    <span
      data-testid={DATA_TEST_ID.VALIDATION_PILL}
      className="inline-flex items-center rounded-full bg-red-100 px-3 py-1 text-xs font-medium text-red-700"
    >
      {t("pipelines.editor.saveBar.issuesCount", { count: issues.length })}
    </span>
  );
}

export function PipelineSaveBar({
  name,
  onNameChange,
  isReadonly,
  issues,
  isValidating,
  isDirty,
  isSaving,
  onSave,
  onCancel,
}: PipelineSaveBarProps) {
  const isSaveDisabled =
    isSaving || issues.length > 0 || !isDirty || isReadonly;

  return (
    <div
      data-testid={DATA_TEST_ID.ROOT}
      className="fixed bottom-0 left-0 right-0 z-[100] flex items-center gap-4 border-t border-line bg-paper px-6 py-3"
    >
      {/* Name input — hidden for readonly pipelines */}
      {!isReadonly && (
        <Input
          data-testid={DATA_TEST_ID.NAME_INPUT}
          value={name}
          disabled={isSaving}
          onChange={(changeEvent) => onNameChange(changeEvent.target.value)}
          className="w-64"
        />
      )}

      {/* Validation pill — centred in the remaining space */}
      <div className="flex flex-1 items-center justify-center">
        <ValidationPill isValidating={isValidating} issues={issues} />
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-2">
        <Button
          data-testid={DATA_TEST_ID.CANCEL_BUTTON}
          variant="ghost"
          disabled={isSaving}
          onClick={onCancel}
        >
          {"Cancel"}
        </Button>
        <Button
          data-testid={DATA_TEST_ID.SAVE_BUTTON}
          variant="primary"
          disabled={isSaveDisabled}
          loading={isSaving}
          onClick={onSave}
        >
          {"Save"}
        </Button>
      </div>
    </div>
  );
}
