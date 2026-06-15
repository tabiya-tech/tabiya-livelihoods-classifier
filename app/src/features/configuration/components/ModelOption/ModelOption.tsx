/**
 * Wraps the RadioCard primitive with Configuration-page-specific affordances:
 * a small badge that shows the model's capacity ("768-dim", "v1.2.0", …) and
 * an optional "recommended" tag.
 *
 * The component is purely presentational — selection lives in the page's
 * draft state via `useUserConfiguration`. To target a ModelOption in tests,
 * use the inner `RADIO_CARD_DATA_TEST_ID.CONTAINER` (every option is a
 * RadioCard button); the suffix / recommended badges have their own testids.
 */

import { useTranslation } from "react-i18next";
import { RadioCard, Tag, type RadioCardProps } from "@/components";

const uniqueId = "5b8e1c3f-d4a2-4b6e-9f7c-3a2d1e5c8b4d";

export const DATA_TEST_ID = {
  SUFFIX_TAG: `model-option-suffix-tag-${uniqueId}`,
  RECOMMENDED_BADGE: `model-option-recommended-badge-${uniqueId}`,
};

export interface ModelOptionProps
  extends Pick<RadioCardProps, "selected" | "onClick" | "title" | "description"> {
  /** Compact secondary label (e.g. "768-dim", "v1.2.0"). Optional. */
  suffix?: string;
  /** Show the "recommended" badge alongside the suffix. */
  recommended?: boolean;
  /**
   * Optional stable identifier surfaced as `data-model-id` on the underlying
   * RadioCard button. Lets the page locate a specific option without depending
   * on its label.
   */
  modelId?: string;
}

export function ModelOption({
  selected,
  onClick,
  title,
  description,
  suffix,
  recommended,
  modelId,
}: ModelOptionProps) {
  const { t } = useTranslation();
  const hasMeta = Boolean(suffix) || Boolean(recommended);
  const meta = hasMeta ? (
    <>
      {suffix && (
        <Tag data-testid={DATA_TEST_ID.SUFFIX_TAG} size="sm">
          {suffix}
        </Tag>
      )}
      {recommended && (
        <Tag
          data-testid={DATA_TEST_ID.RECOMMENDED_BADGE}
          size="sm"
          tone="lime"
        >
          {t("configuration.modelOption.recommendedBadge")}
        </Tag>
      )}
    </>
  ) : undefined;

  return (
    <RadioCard
      selected={selected}
      onClick={onClick}
      title={title}
      description={description}
      meta={meta}
      data-model-id={modelId}
    />
  );
}
