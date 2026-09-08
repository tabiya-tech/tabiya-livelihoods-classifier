/**
 * Two sliders that scope a Classifier run: top_k (how many ESCO matches
 * per entity) and min_similarity (score cutoff). The page wires these
 * through URL params so a result is shareable.
 *
 * Sticky in the source pane — the user can tweak between runs without
 * scrolling away from the input.
 */

import { useTranslation } from "react-i18next";
import { Slider } from "@/components";
import { mergeClassNames } from "@/lib/mergeClassNames";

const uniqueId = "a7c3e1d8-5f9b-4a2c-8e7d-1b3f6a4e2c8d";

export const DATA_TEST_ID = {
  CONTAINER: `pipeline-params-card-container-${uniqueId}`,
  TOP_K_SLIDER: `pipeline-params-card-top-k-slider-${uniqueId}`,
  MIN_SIMILARITY_SLIDER: `pipeline-params-card-min-similarity-slider-${uniqueId}`,
};

export interface PipelineParamsCardProps {
  topK: number;
  minSimilarity: number;
  onTopKChange: (value: number) => void;
  onMinSimilarityChange: (value: number) => void;
  /** Disabled while a run is in flight. */
  disabled?: boolean;
  className?: string;
}

/** Backend caps: 1–50 for top_k, 0.0–1.0 for min_similarity. */
export const TOP_K_MIN = 1;
export const TOP_K_MAX = 50;
export const MIN_SIMILARITY_MIN = 0;
export const MIN_SIMILARITY_MAX = 1;
export const MIN_SIMILARITY_STEP = 0.05;

export function PipelineParamsCard({
  topK,
  minSimilarity,
  onTopKChange,
  onMinSimilarityChange,
  disabled = false,
  className,
}: PipelineParamsCardProps) {
  const { t } = useTranslation();

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        "flex flex-col gap-4 rounded-md border border-line bg-paper px-4 py-3.5",
        className,
      )}
    >
      <Slider
        data-testid={DATA_TEST_ID.TOP_K_SLIDER}
        label={t("classifier.pipelineParams.topKLabel")}
        hint={t("classifier.pipelineParams.topKHint")}
        min={TOP_K_MIN}
        max={TOP_K_MAX}
        step={1}
        value={topK}
        disabled={disabled}
        onChange={(event) => onTopKChange(Number(event.target.value))}
        format={(value) => String(value)}
      />
      <Slider
        data-testid={DATA_TEST_ID.MIN_SIMILARITY_SLIDER}
        label={t("classifier.pipelineParams.minSimilarityLabel")}
        hint={t("classifier.pipelineParams.minSimilarityHint")}
        min={MIN_SIMILARITY_MIN}
        max={MIN_SIMILARITY_MAX}
        step={MIN_SIMILARITY_STEP}
        value={minSimilarity}
        disabled={disabled}
        onChange={(event) =>
          onMinSimilarityChange(Number(event.target.value))
        }
        format={(value) => value.toFixed(2)}
      />
    </div>
  );
}
