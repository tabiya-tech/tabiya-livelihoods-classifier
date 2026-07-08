/**
 * Classifier page — paste a job ad, run /v2/classify, see linked entities.
 *
 * Composition:
 *   - {@link SourcePane} on the left: text input, drag-drop upload,
 *     pipeline params (URL-synced), filter chips, run/clear buttons, and
 *     a "current config" chip linking to /configuration.
 *   - {@link ResultsTabs} on the right: Entities / Table / JSON views over
 *     the latest response.
 *   - {@link EntityDetailDrawer} for the selected entity's ESCO matches.
 *
 * The page owns:
 *   - source text and entity-type filter (local state)
 *   - selected entity index (local state) — drawer + EntityHighlight share it
 *   - active results tab id (local state)
 *   - the toast surface for run failures
 *
 * URL params (top_k, min_sim) live in {@link useClassifierUrlState} for
 * shareable runs. Source text is intentionally NOT in the URL — it can be
 * large and sometimes sensitive.
 */

import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Icon, Tag, useToast } from "@/components";
import type {
  ClassifiedEntity,
  ClassifyEntityType,
  ClassifyRequest,
} from "@/lib/api";
import { routerPaths } from "@/routes/routerPaths";
import { mergeClassNames } from "@/lib/mergeClassNames";
import { useUserConfiguration } from "../../configuration/hooks/useUserConfiguration";
import { EntityDetailDrawer } from "../components/EntityDetailDrawer/EntityDetailDrawer";
import {
  ENTITY_TYPES,
  SourcePane,
} from "../components/SourcePane/SourcePane";
import {
  ResultsTabs,
  type ResultsTabId,
} from "../components/ResultsTabs/ResultsTabs";
import { useClassify } from "../hooks/useClassify";
import { useClassifierUrlState } from "../hooks/useClassifierUrlState";
import { normalizeClassifyInput } from "../lib/normalizeClassifyInput";

const uniqueId = "8d3e1b6c-4f9a-4c5d-9e2f-7a6b3d8c1e4f";

export const DATA_TEST_ID = {
  CONTAINER: `classifier-page-container-${uniqueId}`,
  EYEBROW: `classifier-page-eyebrow-${uniqueId}`,
  TITLE: `classifier-page-title-${uniqueId}`,
  INTRO: `classifier-page-intro-${uniqueId}`,
  CONFIG_CHIP: `classifier-page-config-chip-${uniqueId}`,
  CONFIG_LINK: `classifier-page-config-link-${uniqueId}`,
};

export function ClassifierPage() {
  const { t } = useTranslation();
  const toast = useToast();

  // ── Local state ──────────────────────────────────────────────────────
  const [text, setText] = useState("");
  const [selectedEntityTypes, setSelectedEntityTypes] = useState<
    Set<ClassifyEntityType>
  >(() => new Set(ENTITY_TYPES));
  const [selectedEntityIndex, setSelectedEntityIndex] = useState<number | null>(
    null,
  );
  const [activeTabId, setActiveTabId] = useState<ResultsTabId>("entities");

  // ── URL-synced params ────────────────────────────────────────────────
  const { topK, minSimilarity, setTopK, setMinSimilarity } =
    useClassifierUrlState();

  // ── Backend hooks ────────────────────────────────────────────────────
  const classifyState = useClassify();
  const configuration = useUserConfiguration();

  const isRunning = classifyState.status === "running";
  const response = classifyState.response;
  const entities = response?.entities ?? null;

  async function handleRun() {
    // Normalise before sending so the NER model sees prose-style text. The
    // returned entity spans are offsets into the normalised string, so we
    // also adopt it as the text the source pane renders highlights against.
    const normalised = normalizeClassifyInput(text);
    if (!normalised) return;
    if (normalised !== text) setText(normalised);
    const payload: ClassifyRequest = {
      text: normalised,
      options: { top_k: topK, min_similarity: minSimilarity },
    };
    setSelectedEntityIndex(null);
    try {
      await classifyState.run(payload);
    } catch {
      toast.show({
        message: t("classifier.toasts.runError"),
        tone: "error",
      });
    }
  }

  function handleClear() {
    setText("");
    setSelectedEntityIndex(null);
    classifyState.reset();
  }

  function handleEntitySelect(index: number | null) {
    setSelectedEntityIndex(index);
  }

  function handleEntityClickFromResults(
    _entity: ClassifiedEntity,
    entityIndex: number,
  ) {
    setSelectedEntityIndex(entityIndex);
  }

  // Drop selection if the entity no longer exists in the latest response.
  useEffect(() => {
    if (selectedEntityIndex === null) return;
    if (!entities) return;
    if (selectedEntityIndex >= entities.length) {
      setSelectedEntityIndex(null);
    }
  }, [entities, selectedEntityIndex]);

  const selectedEntity =
    entities != null && selectedEntityIndex !== null
      ? (entities[selectedEntityIndex] ?? null)
      : null;

  const canRun = text.trim().length > 0;

  const activeConfigChip = useMemo(
    () => <ActiveConfigChip configuration={configuration} />,
    [configuration],
  );

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className="mx-auto flex h-full min-h-0 w-full max-w-[1400px] flex-col gap-6 overflow-hidden px-8 py-8"
    >
      <header className="flex flex-col gap-2">
        <span data-testid={DATA_TEST_ID.EYEBROW} className="eyebrow">
          {t("classifier.eyebrow")}
        </span>
        <h1 data-testid={DATA_TEST_ID.TITLE} className="h-page m-0">
          {t("classifier.title")}
        </h1>
        <p
          data-testid={DATA_TEST_ID.INTRO}
          className="m-0 max-w-[760px] text-sm leading-relaxed text-muted"
        >
          {t("classifier.intro")}
        </p>
      </header>

      <div className="grid min-h-0 flex-1 gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <SourcePane
          className="min-h-0"
          text={text}
          onTextChange={setText}
          entities={entities}
          selectedEntityTypes={selectedEntityTypes}
          onSelectedEntityTypesChange={setSelectedEntityTypes}
          topK={topK}
          minSimilarity={minSimilarity}
          onTopKChange={setTopK}
          onMinSimilarityChange={setMinSimilarity}
          isRunning={isRunning}
          canRun={canRun}
          onRun={() => {
            void handleRun();
          }}
          onClear={handleClear}
          selectedEntityIndex={selectedEntityIndex}
          onEntitySelect={handleEntitySelect}
          activeConfigSlot={activeConfigChip}
        />

        <aside className="flex min-h-0 flex-col gap-4">
          {response ? (
            <ResultsTabs
              className="min-h-0 flex-1"
              response={response}
              selectedEntityTypes={selectedEntityTypes}
              activeTabId={activeTabId}
              onActiveTabChange={setActiveTabId}
              selectedEntityIndex={selectedEntityIndex}
              onEntityClick={handleEntityClickFromResults}
            />
          ) : (
            <PlaceholderPanel isRunning={isRunning} />
          )}
        </aside>
      </div>

      <EntityDetailDrawer
        open={selectedEntity !== null}
        entity={selectedEntity}
        onClose={() => setSelectedEntityIndex(null)}
      />
    </div>
  );
}

interface PlaceholderPanelProps {
  isRunning: boolean;
}

function PlaceholderPanel({ isRunning }: PlaceholderPanelProps) {
  const { t } = useTranslation();
  return (
    <div
      className={mergeClassNames(
        "flex min-h-0 flex-1 flex-col items-center justify-center gap-2",
        "rounded-md border border-dashed border-line bg-paper px-8 py-12 text-center",
      )}
    >
      <Icon name="spark" size={20} />
      <p className="m-0 font-mono text-xs text-navy">
        {isRunning
          ? t("classifier.placeholder.runningTitle")
          : t("classifier.placeholder.idleTitle")}
      </p>
      <p className="m-0 max-w-[280px] text-xs text-muted">
        {isRunning
          ? t("classifier.placeholder.runningHelp")
          : t("classifier.placeholder.idleHelp")}
      </p>
    </div>
  );
}

interface ActiveConfigChipProps {
  configuration: ReturnType<typeof useUserConfiguration>;
}

function ActiveConfigChip({ configuration }: ActiveConfigChipProps) {
  const { t } = useTranslation();

  const summary = (() => {
    if (configuration.loadStatus !== "ready" || !configuration.saved) {
      return t("classifier.activeConfig.loading");
    }
    return t("classifier.activeConfig.summary", {
      nel: configuration.saved.nel_model_id,
      taxonomy: configuration.saved.taxonomy_model_id,
    });
  })();

  return (
    <div
      data-testid={DATA_TEST_ID.CONFIG_CHIP}
      className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-line bg-paper px-3.5 py-2"
    >
      <span className="inline-flex items-center gap-2 font-mono text-[11px] text-muted">
        <Tag size="sm">
          {t("classifier.activeConfig.tag")}
        </Tag>
        {summary}
      </span>
      <Link
        data-testid={DATA_TEST_ID.CONFIG_LINK}
        to={routerPaths.CONFIGURATION}
        className="inline-flex items-center gap-1 font-mono text-[11px] text-navy underline decoration-line-strong underline-offset-2 hover:decoration-navy"
      >
        {t("classifier.activeConfig.configureLink")}
        <Icon name="arrowRight" size={12} />
      </Link>
    </div>
  );
}
