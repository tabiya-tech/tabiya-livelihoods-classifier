/**
 * A page title that doubles as its own rename field.
 *
 * Renders as plain title text (styled like the surrounding <h1>). Clicking it
 * swaps to a borderless, background-less input sized like the text, so editing
 * looks identical to reading — no visible textbox chrome. Enter / blur commit,
 * Escape reverts. Read-only mode renders static text with no affordance.
 */

import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

const uniqueId = "e3b1c9a7-2d4f-4e6a-9b8c-1f0a2b3c4d5e";

export const DATA_TEST_ID = {
  ROOT: `inline-editable-title-${uniqueId}`,
  DISPLAY: `inline-editable-title-display-${uniqueId}`,
  INPUT: `inline-editable-title-input-${uniqueId}`,
};

export interface InlineEditableTitleProps {
  value: string;
  onChange: (nextValue: string) => void;
  /** Shown greyed when `value` is empty. */
  placeholder?: string;
  /** When true, renders static text with no click-to-edit affordance. */
  readOnly?: boolean;
  className?: string;
}

export function InlineEditableTitle({
  value,
  onChange,
  placeholder: placeholderProp,
  readOnly = false,
  className = "h-page m-0",
}: InlineEditableTitleProps) {
  const { t } = useTranslation();
  const placeholder = placeholderProp ?? t("pipelines.editor.titlePlaceholder");
  const [isEditing, setIsEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);

  // Keep the draft in sync when the value changes from outside while not editing.
  useEffect(() => {
    if (!isEditing) {
      setDraft(value);
    }
  }, [value, isEditing]);

  // Focus + select the text when entering edit mode.
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  function commit() {
    setIsEditing(false);
    const trimmed = draft.trim();
    if (trimmed !== value) {
      onChange(trimmed);
    }
  }

  function cancel() {
    setDraft(value);
    setIsEditing(false);
  }

  const displayText = value || placeholder;
  const isPlaceholder = !value;

  if (readOnly) {
    return (
      <h1 data-testid={DATA_TEST_ID.ROOT} className={className}>
        {displayText}
      </h1>
    );
  }

  if (isEditing) {
    return (
      <input
        ref={inputRef}
        data-testid={DATA_TEST_ID.INPUT}
        className={className}
        value={draft}
        placeholder={placeholder}
        onChange={(changeEvent) => setDraft(changeEvent.target.value)}
        onBlur={commit}
        onKeyDown={(keyEvent) => {
          if (keyEvent.key === "Enter") {
            keyEvent.preventDefault();
            commit();
          } else if (keyEvent.key === "Escape") {
            keyEvent.preventDefault();
            cancel();
          }
        }}
        // Borderless / background-less so it looks exactly like the heading.
        style={{
          border: "none",
          outline: "none",
          background: "transparent",
          padding: 0,
          margin: 0,
          font: "inherit",
          color: "inherit",
          width: "100%",
          minWidth: "200px",
        }}
      />
    );
  }

  return (
    <h1
      data-testid={DATA_TEST_ID.ROOT}
      className={className}
      role="button"
      tabIndex={0}
      title={t("pipelines.editor.titleClickToRename")}
      onClick={() => setIsEditing(true)}
      onKeyDown={(keyEvent) => {
        if (keyEvent.key === "Enter" || keyEvent.key === " ") {
          keyEvent.preventDefault();
          setIsEditing(true);
        }
      }}
      style={{
        cursor: "text",
        color: isPlaceholder ? "#8a8780" : undefined,
      }}
    >
      <span data-testid={DATA_TEST_ID.DISPLAY}>{displayText}</span>
    </h1>
  );
}
