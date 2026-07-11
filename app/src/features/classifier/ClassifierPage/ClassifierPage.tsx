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
import { useTranslation } from "react-i18next";
import { useToast } from "@/components";
import type {
  ClassifiedEntity,
  ClassifyEntityType,
  ClassifyRequest,
} from "@/lib/api";
import { mergeClassNames } from "@/lib/mergeClassNames";
import { EntityDetailDrawer } from "../components/EntityDetailDrawer/EntityDetailDrawer";
import { PipelineSelectorChip, PipelineStageChips } from "../components/PipelineSelectorChip/PipelineSelectorChip";
import {
  ENTITY_TYPES,
  SourcePane,
} from "../components/SourcePane/SourcePane";
import {
  ResultsTabs,
  type ResultsTabId,
} from "../components/ResultsTabs/ResultsTabs";
import { useActivePipeline } from "../hooks/useActivePipeline";
import { useClassify } from "../hooks/useClassify";
import { useClassifierUrlState } from "../hooks/useClassifierUrlState";
import { usePipelineOutputSlot } from "../hooks/usePipelineOutputSlot";
import { normalizeClassifyInput } from "../lib/normalizeClassifyInput";
import { defaultsFromPipeline } from "../lib/defaultsFromPipeline";

const uniqueId = "8d3e1b6c-4f9a-4c5d-9e2f-7a6b3d8c1e4f";

export const DATA_TEST_ID = {
  CONTAINER: `classifier-page-container-${uniqueId}`,
  EYEBROW: `classifier-page-eyebrow-${uniqueId}`,
  TITLE: `classifier-page-title-${uniqueId}`,
  INTRO: `classifier-page-intro-${uniqueId}`,
  PIPELINE_SELECTOR_SLOT: `classifier-page-pipeline-selector-slot-${uniqueId}`,
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
  const activePipelineState = useActivePipeline();
  const pipelineOutputSlot = usePipelineOutputSlot(
    activePipelineState.activePipeline,
  );
  const showMatches = pipelineOutputSlot.outputSlotType !== "Entities";

  const isRunning = classifyState.status === "running";
  const response = classifyState.response;
  const entities = response?.entities ?? null;

  // Seed entity-type filter from the active pipeline's NER stage config
  // whenever the pipeline changes. Falls back to all types when the pipeline
  // has no entity_types config.
  useEffect(() => {
    const defaults = defaultsFromPipeline(activePipelineState.activePipeline ?? null);
    setSelectedEntityTypes(
      new Set(defaults.entityTypes ?? ENTITY_TYPES),
    );
  }, [activePipelineState.activePipeline]);

  async function handleRun() {
    // Normalise before sending so the NER model sees prose-style text. The
    // returned entity spans are offsets into the normalised string, so we
    // also adopt it as the text the source pane renders highlights against.
    const normalised = normalizeClassifyInput(text);
    if (!normalised) return;
    if (normalised !== text) setText(normalised);
    const extractEntities = [...selectedEntityTypes] as ClassifyEntityType[];
    const payload: ClassifyRequest = {
      text: normalised,
      pipeline_id: activePipelineState.activePipeline?.pipeline_id,
      options: {
        top_k: topK,
        min_similarity: minSimilarity,
        extract_entities: extractEntities.length < ENTITY_TYPES.length ? extractEntities : undefined,
      },
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

  async function handlePipelineChange(pipelineId: string) {
    try {
      await activePipelineState.setActivePipeline(pipelineId);
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

  const pipelineSelectorSlot = useMemo(
    () => (
      <div
        data-testid={DATA_TEST_ID.PIPELINE_SELECTOR_SLOT}
        className="flex flex-col gap-2 sm:flex-row sm:items-center"
      >
        <PipelineSelectorChip
          pipelines={activePipelineState.pipelines}
          selectedPipelineId={
            activePipelineState.activePipeline?.pipeline_id ?? null
          }
          onPipelineChange={(pipelineId) => {
            void handlePipelineChange(pipelineId);
          }}
          isLoading={activePipelineState.status === "loading"}
        />
        {activePipelineState.activePipeline &&
          activePipelineState.activePipeline.stages.length > 0 && (
            <PipelineStageChips
              stages={activePipelineState.activePipeline.stages}
            />
          )}
      </div>
    ),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      activePipelineState.pipelines,
      activePipelineState.activePipeline,
      activePipelineState.status,
    ],
  );

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className="mx-auto flex h-full min-h-0 w-full max-w-[1400px] flex-col gap-6 px-4 py-6 md:px-8 md:py-8"
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

      <div className="grid gap-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <SourcePane
          className=""
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
          activeConfigSlot={pipelineSelectorSlot}
        />

        <aside className="flex flex-col gap-4">
          {response ? (
            <ResultsTabs
              className=""
              response={response}
              selectedEntityTypes={selectedEntityTypes}
              activeTabId={activeTabId}
              onActiveTabChange={setActiveTabId}
              selectedEntityIndex={selectedEntityIndex}
              onEntityClick={showMatches ? handleEntityClickFromResults : undefined}
              showMatches={showMatches}
            />
          ) : (
            <PlaceholderPanel isRunning={isRunning} />
          )}
        </aside>
      </div>

      {showMatches && (
        <EntityDetailDrawer
          open={selectedEntity !== null}
          entity={selectedEntity}
          onClose={() => setSelectedEntityIndex(null)}
        />
      )}
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

