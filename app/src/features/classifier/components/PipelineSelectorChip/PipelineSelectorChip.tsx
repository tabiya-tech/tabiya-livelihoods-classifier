/**
 * Compact "which pipeline is running?" chip on the Classifier page.
 *
 * Clicking the chip opens a popover with every pipeline the caller owns
 * plus a link back to /pipelines for edits. Picking a row fires
 * onPipelineChange — the parent hook (`useActivePipeline`) does the
 * optimistic switch and the actual /v2/pipelines/{id}/activate call.
 *
 * The popover is deliberately unstyled beyond the basics — no external
 * dropdown library. It closes on: outside click, Escape, or selecting a
 * row (the caller decides whether the click callback also closes it, so
 * the chip can debounce or animate independently if needed).
 */

import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
} from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Icon, Tag } from "@/components";
import type { Pipeline } from "@/lib/api";
import { mergeClassNames } from "@/lib/mergeClassNames";
import { routerPaths } from "@/routes/routerPaths";

const uniqueId = "e2a1c9d4-7b8f-4d3a-a6c5-1f9b8e2d7c4a";

export const DATA_TEST_ID = {
  ROOT: `pipeline-selector-chip-root-${uniqueId}`,
  TRIGGER: `pipeline-selector-chip-trigger-${uniqueId}`,
  POPOVER: `pipeline-selector-chip-popover-${uniqueId}`,
  PIPELINE_ROW: `pipeline-selector-chip-pipeline-row-${uniqueId}`,
  EDIT_LINK: `pipeline-selector-chip-edit-link-${uniqueId}`,
};

export interface PipelineSelectorChipProps {
  /** Pipelines the user can pick from. */
  pipelines: Pipeline[];
  /** The currently-selected pipeline (usually the active one), or null while loading. */
  selectedPipelineId: string | null;
  /** Fires when the user picks a different pipeline from the popover. */
  onPipelineChange: (pipelineId: string) => void;
  /** True while `pipelines` is still loading. */
  isLoading: boolean;
  /** Force-open the popover — used by stories to render the open state. */
  defaultOpen?: boolean;
  className?: string;
}

export function PipelineSelectorChip({
  pipelines,
  selectedPipelineId,
  onPipelineChange,
  isLoading,
  defaultOpen = false,
  className,
}: PipelineSelectorChipProps) {
  const { t } = useTranslation();
  const rootRef = useRef<HTMLDivElement | null>(null);
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const popoverId = useId();

  const selectedPipeline =
    pipelines.find(
      (pipeline) => pipeline.pipeline_id === selectedPipelineId,
    ) ?? null;

  const triggerLabel = (() => {
    if (isLoading) return t("classifier.pipelineSelector.loading");
    if (selectedPipeline) return selectedPipeline.name;
    return t("classifier.pipelineSelector.loading");
  })();

  // Close on outside click. We compare against rootRef so the popover
  // (which is a child of root) still counts as "inside".
  useEffect(() => {
    if (!isOpen) return;
    function handleClickOutside(event: MouseEvent) {
      if (!rootRef.current) return;
      if (rootRef.current.contains(event.target as Node)) return;
      setIsOpen(false);
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setIsOpen(false);
    }
    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen]);

  const handleTriggerClick = useCallback(() => {
    if (isLoading) return;
    setIsOpen((previous) => !previous);
  }, [isLoading]);

  const handleRowClick = useCallback(
    (pipelineId: string) => {
      onPipelineChange(pipelineId);
      setIsOpen(false);
    },
    [onPipelineChange],
  );

  return (
    <div
      ref={rootRef}
      data-testid={DATA_TEST_ID.ROOT}
      className={mergeClassNames("relative inline-flex flex-col", className)}
    >
      <button
        type="button"
        data-testid={DATA_TEST_ID.TRIGGER}
        aria-label={t("classifier.pipelineSelector.ariaLabel")}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={popoverId}
        onClick={handleTriggerClick}
        disabled={isLoading}
        className={mergeClassNames(
          "inline-flex items-center justify-between gap-2 rounded-md border border-line bg-paper px-3.5 py-2",
          "font-mono text-[11px] text-navy",
          "transition-colors hover:border-line-strong disabled:cursor-not-allowed disabled:opacity-60",
        )}
      >
        <span className="inline-flex items-center gap-2">
          <Tag size="sm">{triggerLabel}</Tag>
        </span>
        <Icon
          name="arrowRight"
          size={12}
          className={mergeClassNames(
            "transition-transform",
            isOpen ? "-rotate-90" : "rotate-90",
          )}
        />
      </button>

      {isOpen && (
        <div
          id={popoverId}
          role="listbox"
          data-testid={DATA_TEST_ID.POPOVER}
          className={mergeClassNames(
            "absolute left-0 top-full z-20 mt-2 flex w-full min-w-[240px] flex-col",
            "rounded-md border border-line bg-paper shadow-lg",
          )}
        >
          <ul className="flex flex-col py-1">
            {pipelines.map((pipeline) => {
              const isSelected =
                pipeline.pipeline_id === selectedPipelineId;
              return (
                <li key={pipeline.pipeline_id}>
                  <button
                    type="button"
                    data-testid={DATA_TEST_ID.PIPELINE_ROW}
                    data-pipeline-id={pipeline.pipeline_id}
                    role="option"
                    aria-selected={isSelected}
                    onClick={() => handleRowClick(pipeline.pipeline_id)}
                    className={mergeClassNames(
                      "flex w-full items-center justify-between gap-3 px-3 py-2 text-left",
                      "font-mono text-[11px] text-navy",
                      "hover:bg-line/40",
                    )}
                  >
                    <span className="truncate">{pipeline.name}</span>
                    {isSelected && (
                      <span className="inline-flex items-center gap-1">
                        <Icon
                          name="check"
                          size={12}
                          aria-hidden
                        />
                        <Tag size="sm" tone="lime">
                          {t("classifier.pipelineSelector.activeBadge")}
                        </Tag>
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>
          <div className="border-t border-line px-3 py-2">
            <Link
              data-testid={DATA_TEST_ID.EDIT_LINK}
              to={routerPaths.PIPELINES}
              className="inline-flex items-center gap-1 font-mono text-[11px] text-navy underline decoration-line-strong underline-offset-2 hover:decoration-navy"
            >
              {t("classifier.pipelineSelector.editLink")}
              <Icon name="arrowRight" size={12} />
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
