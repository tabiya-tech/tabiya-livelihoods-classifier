/**
 * Drop a `.txt` (or other plain-text) file onto this region — or click to
 * pick — and the parent gets the file's contents as a string.
 *
 * Pure presentation + a thin FileReader wrapper. Network never enters the
 * picture; the page submits the text via `useClassify` separately.
 */

import { useRef, useState, type DragEvent } from "react";
import { useTranslation } from "react-i18next";
import { Icon } from "@/components";
import { mergeClassNames } from "@/lib/mergeClassNames";

const uniqueId = "9c4b6e2d-3f8a-4d7c-9e1b-5a8c2d6f4b3e";

export const DATA_TEST_ID = {
  CONTAINER: `upload-dropzone-container-${uniqueId}`,
  FILE_INPUT: `upload-dropzone-file-input-${uniqueId}`,
  PROMPT: `upload-dropzone-prompt-${uniqueId}`,
  ERROR: `upload-dropzone-error-${uniqueId}`,
};

const DEFAULT_ACCEPT = ".txt,text/plain,.md,text/markdown,application/json";
/** 1 MB — big enough for any real job ad, small enough to refuse abuse. */
export const DEFAULT_MAX_BYTES = 1_048_576;

export interface UploadDropzoneProps {
  /** Fires with the decoded text once a file is dropped or picked. */
  onText: (text: string, filename: string) => void;
  /** Comma-separated `accept` value. Defaults to plain-text-ish types. */
  accept?: string;
  /** Bytes — files larger than this are rejected with a localized error. */
  maxBytes?: number;
  /** Disabled while a run is in flight. */
  disabled?: boolean;
  className?: string;
}

export function UploadDropzone({
  onText,
  accept = DEFAULT_ACCEPT,
  maxBytes = DEFAULT_MAX_BYTES,
  disabled = false,
  className,
}: UploadDropzoneProps) {
  const { t } = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFile(file: File) {
    setError(null);
    if (file.size > maxBytes) {
      setError(
        t("classifier.upload.errorTooLarge", {
          maxKb: Math.floor(maxBytes / 1024),
        }),
      );
      return;
    }
    try {
      const text = await file.text();
      onText(text, file.name);
    } catch (caught) {
      setError(t("classifier.upload.errorRead"));
      // eslint-disable-next-line no-console
      console.error("Upload read failed", caught);
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragOver(false);
    if (disabled) return;
    const file = event.dataTransfer.files?.[0];
    if (file) void handleFile(file);
  }

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setIsDragOver(true);
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDrop}
      onClick={() => {
        if (!disabled) inputRef.current?.click();
      }}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      onKeyDown={(event) => {
        if (disabled) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          inputRef.current?.click();
        }
      }}
      className={mergeClassNames(
        "flex cursor-pointer flex-col items-center justify-center gap-2",
        "rounded-md border border-dashed px-4 py-6 text-center transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy/30",
        isDragOver
          ? "border-navy bg-cream-200 text-navy"
          : "border-line bg-paper text-muted hover:border-line-strong",
        disabled && "cursor-not-allowed opacity-45",
        className,
      )}
    >
      <Icon name="upload" size={20} />
      <span
        data-testid={DATA_TEST_ID.PROMPT}
        className="font-mono text-xs text-navy"
      >
        {t("classifier.upload.prompt")}
      </span>
      <span className="text-[11px] text-muted-2">
        {t("classifier.upload.subPrompt")}
      </span>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        data-testid={DATA_TEST_ID.FILE_INPUT}
        className="sr-only"
        disabled={disabled}
        onClick={(event) => event.stopPropagation()}
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void handleFile(file);
          event.target.value = "";
        }}
      />
      {error && (
        <span
          data-testid={DATA_TEST_ID.ERROR}
          className="font-mono text-[11px] text-error"
        >
          {error}
        </span>
      )}
    </div>
  );
}
