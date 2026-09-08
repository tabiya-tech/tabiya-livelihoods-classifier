/**
 * UI-only preview bar for the pipeline editor.
 *
 * Walks a synthetic "active stage" pointer across the pipeline stages on a
 * fixed interval. The animation is entirely local — this component NEVER
 * calls the backend and MUST NOT import from `@/lib/api`. That contract is
 * verified by test to keep the "preview" affordance clearly separate from
 * the (future) real /v2/classify run.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components";

const uniqueId = "d5e6f7a8-b9c0-4d1e-9f2a-3b4c5d6e7f80";

export const DATA_TEST_ID = {
  ROOT: `run-preview-bar-root-${uniqueId}`,
  PLAY_BUTTON: `run-preview-bar-play-button-${uniqueId}`,
  RESET_BUTTON: `run-preview-bar-reset-button-${uniqueId}`,
  STATUS_LABEL: `run-preview-bar-status-label-${uniqueId}`,
};

const DEFAULT_STEP_INTERVAL_MS = 600;

export interface RunPreviewBarProps {
  /** Number of stages in the pipeline; determines how many steps the animation walks through. */
  stageCount: number;
  /** Fires whenever the active node index changes (including Reset → null). */
  onActiveStageChange: (activeStageIndex: number | null) => void;
  /** Milliseconds between stage advances. Default 600ms. */
  stepIntervalMs?: number;
  className?: string;
}

export function RunPreviewBar({
  stageCount,
  onActiveStageChange,
  stepIntervalMs = DEFAULT_STEP_INTERVAL_MS,
  className,
}: RunPreviewBarProps) {
  const { t } = useTranslation();

  const [activeStageIndex, setActiveStageIndex] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  // Hold the pending timeout so both Reset and unmount can clear it.
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearPendingTimeout = useCallback(() => {
    if (timeoutRef.current !== null) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  useEffect(() => {
    // On unmount, drop any pending step. This never fires during normal
    // reset/play cycles — clearPendingTimeout inside handlers covers those.
    return () => {
      if (timeoutRef.current !== null) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, []);

  const scheduleNextStep = useCallback(
    (nextIndex: number) => {
      timeoutRef.current = setTimeout(() => {
        if (nextIndex > stageCount - 1) {
          setIsPlaying(false);
          timeoutRef.current = null;
          return;
        }
        setActiveStageIndex(nextIndex);
        onActiveStageChange(nextIndex);
        scheduleNextStep(nextIndex + 1);
      }, stepIntervalMs);
    },
    [onActiveStageChange, stageCount, stepIntervalMs],
  );

  function handlePlay() {
    if (stageCount === 0 || isPlaying) return;
    clearPendingTimeout();
    setIsPlaying(true);
    setActiveStageIndex(0);
    onActiveStageChange(0);
    if (stageCount === 1) {
      // Nothing to advance to; end immediately at the last (only) stage.
      setIsPlaying(false);
      return;
    }
    scheduleNextStep(1);
  }

  function handleReset() {
    clearPendingTimeout();
    setIsPlaying(false);
    setActiveStageIndex(null);
    onActiveStageChange(null);
  }

  const isPlayDisabled = stageCount === 0 || isPlaying;
  const isResetDisabled = !isPlaying && activeStageIndex === null;

  const statusText = isPlaying
    ? t("pipelines.editor.runPreview.status.playing")
    : activeStageIndex !== null
      ? t("pipelines.editor.runPreview.status.done")
      : t("pipelines.editor.runPreview.status.idle");

  return (
    <div
      data-testid={DATA_TEST_ID.ROOT}
      className={
        className ??
        "flex items-center gap-3 rounded-md border border-line bg-paper px-3 py-2"
      }
    >
      <Button
        data-testid={DATA_TEST_ID.PLAY_BUTTON}
        variant="primary"
        size="sm"
        disabled={isPlayDisabled}
        onClick={handlePlay}
      >
        {t("pipelines.editor.runPreview.playButton")}
      </Button>
      <Button
        data-testid={DATA_TEST_ID.RESET_BUTTON}
        variant="ghost"
        size="sm"
        disabled={isResetDisabled}
        onClick={handleReset}
      >
        {t("pipelines.editor.runPreview.resetButton")}
      </Button>
      <span
        data-testid={DATA_TEST_ID.STATUS_LABEL}
        className="font-mono text-xs text-muted"
      >
        {statusText}
      </span>
    </div>
  );
}
