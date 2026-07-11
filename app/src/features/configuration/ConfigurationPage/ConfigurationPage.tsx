/**
 * Configuration page — pick the models that power classifications.
 *
 * Composes:
 * - {@link StageRail} on the left (one entry per stage: NEL, Taxonomy).
 * - The right panel renders the active stage's model options as a radio
 *   group of {@link ModelOption}s. Selections mutate the in-flight draft
 *   from {@link useUserConfiguration}.
 * - {@link SaveBar} sits at the bottom; visible while the draft is dirty
 *   or the most-recent save is still showing its "saved" indicator.
 * - {@link UnsavedChangesGuard} intercepts any in-app navigation while the
 *   draft is dirty via the app's {@link NavigationGuardProvider}.
 *
 * The page owns the orchestration; every visual piece is a leaf component.
 */

import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { EmptyState, Spinner, useToast } from "@/components";
import type { V2UserConfig } from "@/lib/api";
import { ModelOption } from "../components/ModelOption/ModelOption";
import { SaveBar } from "../components/SaveBar/SaveBar";
import {
  StageRail,
  type StageRailItem,
} from "../components/StageRail/StageRail";
import { UnsavedChangesGuard } from "../components/UnsavedChangesGuard/UnsavedChangesGuard";
import { useNelModels } from "../hooks/useNelModels";
import { useTaxonomyModels } from "../hooks/useTaxonomyModels";
import { useUnsavedChangesGuard } from "../hooks/useUnsavedChangesGuard";
import { useUserConfiguration } from "../hooks/useUserConfiguration";

const uniqueId = "7d2e9f4b-1c8a-4e5d-9b3f-6a1c4e8d2b5f";

export const DATA_TEST_ID = {
  CONTAINER: `configuration-page-container-${uniqueId}`,
  EYEBROW: `configuration-page-eyebrow-${uniqueId}`,
  TITLE: `configuration-page-title-${uniqueId}`,
  INTRO: `configuration-page-intro-${uniqueId}`,
  PANEL: `configuration-page-panel-${uniqueId}`,
  PANEL_EYEBROW: `configuration-page-panel-eyebrow-${uniqueId}`,
  PANEL_TITLE: `configuration-page-panel-title-${uniqueId}`,
  PANEL_INTRO: `configuration-page-panel-intro-${uniqueId}`,
  SECTION_LABEL: `configuration-page-section-label-${uniqueId}`,
  LOADING: `configuration-page-loading-${uniqueId}`,
  LOAD_ERROR: `configuration-page-load-error-${uniqueId}`,
  OPTION_LIST: `configuration-page-option-list-${uniqueId}`,
};

type StageId = "nel" | "taxonomy";

/**
 * The model id we tag as "recommended" in each stage's option list.
 * Matches the fixture's default selection.
 */
const RECOMMENDED_NEL_MODEL_ID = "mpnet-base-v2";
const RECOMMENDED_TAXONOMY_MODEL_ID = "esco-1.2.0";

export function ConfigurationPage() {
  const { t } = useTranslation();
  const toast = useToast();
  const nelModels = useNelModels();
  const taxonomyModels = useTaxonomyModels();
  const configuration = useUserConfiguration();

  const [activeStageId, setActiveStageId] = useState<StageId>("nel");

  const guard = useUnsavedChangesGuard(configuration.isDirty);

  const nelModelById = useMemo(
    () => new Map(nelModels.models.map((model) => [model.model_id, model])),
    [nelModels.models],
  );
  const taxonomyModelById = useMemo(
    () => new Map(taxonomyModels.models.map((model) => [model.id, model])),
    [taxonomyModels.models],
  );

  const stageItems: StageRailItem<StageId>[] = [
    {
      id: "nel",
      number: t("configuration.stages.nel.number"),
      label: t("configuration.stages.nel.label"),
      subLabel: t("configuration.stages.nel.subLabel"),
      currentValue: configuration.saved
        ? (nelModelById.get(configuration.saved.nel_model_id)?.model_id ??
          configuration.saved.nel_model_id)
        : undefined,
    },
    {
      id: "taxonomy",
      number: t("configuration.stages.taxonomy.number"),
      label: t("configuration.stages.taxonomy.label"),
      subLabel: t("configuration.stages.taxonomy.subLabel"),
      currentValue: configuration.saved
        ? (() => {
            const taxonomy = taxonomyModelById.get(
              configuration.saved.taxonomy_model_id,
            );
            return taxonomy
              ? `${taxonomy.name} ${taxonomy.version}`
              : configuration.saved.taxonomy_model_id;
          })()
        : undefined,
    },
  ];

  async function handleSave() {
    try {
      await configuration.save();
    } catch {
      toast.show({
        message: t("configuration.toasts.saveError"),
        tone: "error",
      });
    }
  }

  function handleNelSelect(modelId: string) {
    const patch: Partial<V2UserConfig> = { nel_model_id: modelId };
    configuration.setDraft(patch);
  }

  function handleTaxonomySelect(taxonomyId: string) {
    const patch: Partial<V2UserConfig> = { taxonomy_model_id: taxonomyId };
    configuration.setDraft(patch);
  }

  const isLoading =
    configuration.loadStatus === "loading" ||
    nelModels.status === "loading" ||
    taxonomyModels.status === "loading";

  const loadError =
    configuration.loadError ?? nelModels.error ?? taxonomyModels.error;

  const activeDraft = configuration.draft;

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className="mx-auto flex w-full max-w-[1080px] flex-col gap-8 px-4 py-6 sm:px-8 sm:py-10"
    >
      <header className="flex flex-col gap-2">
        <span data-testid={DATA_TEST_ID.EYEBROW} className="eyebrow">
          {t("configuration.eyebrow")}
        </span>
        <h1 data-testid={DATA_TEST_ID.TITLE} className="h-page m-0">
          {t("configuration.title")}
        </h1>
        <p
          data-testid={DATA_TEST_ID.INTRO}
          className="m-0 max-w-[680px] text-sm leading-relaxed text-muted"
        >
          {t("configuration.intro")}
        </p>
      </header>

      {isLoading && (
        <div
          data-testid={DATA_TEST_ID.LOADING}
          className="flex items-center gap-2 rounded-md border border-line bg-paper px-4 py-3 font-mono text-xs text-muted"
        >
          <Spinner />
          {t("common.loading")}
        </div>
      )}

      {!isLoading && loadError && (
        <div data-testid={DATA_TEST_ID.LOAD_ERROR}>
          <EmptyState
            icon="close"
            title={t("configuration.errors.loadFailed")}
          />
        </div>
      )}

      {!isLoading && !loadError && activeDraft && (
        <div className="grid grid-cols-[240px_1fr] gap-8">
          <StageRail
            items={stageItems}
            activeId={activeStageId}
            onSelect={setActiveStageId}
            aria-label={t("configuration.title")}
          />

          <section
            data-testid={DATA_TEST_ID.PANEL}
            className="flex flex-col gap-5"
          >
            {activeStageId === "nel" && (
              <>
                <PanelHeader
                  eyebrow={t("configuration.stages.nel.panelEyebrow")}
                  title={t("configuration.stages.nel.panelTitle")}
                  intro={t("configuration.stages.nel.panelIntro")}
                />
                <div
                  data-testid={DATA_TEST_ID.OPTION_LIST}
                  role="radiogroup"
                  aria-labelledby={DATA_TEST_ID.SECTION_LABEL}
                  className="flex flex-col gap-2"
                >
                  <span
                    id={DATA_TEST_ID.SECTION_LABEL}
                    data-testid={DATA_TEST_ID.SECTION_LABEL}
                    className="font-mono text-[11px] uppercase tracking-[0.08em] text-muted-2"
                  >
                    {t("configuration.stages.nel.sectionLabel")}
                  </span>
                  {nelModels.models.map((model) => (
                    <ModelOption
                      key={model.model_id}
                      modelId={model.model_id}
                      title={model.model_id}
                      description={model.description}
                      suffix={t("configuration.modelOption.dimensionsSuffix", {
                        dimensions: model.dimensions,
                      })}
                      recommended={model.model_id === RECOMMENDED_NEL_MODEL_ID}
                      selected={activeDraft.nel_model_id === model.model_id}
                      onClick={() => handleNelSelect(model.model_id)}
                    />
                  ))}
                </div>
              </>
            )}

            {activeStageId === "taxonomy" && (
              <>
                <PanelHeader
                  eyebrow={t("configuration.stages.taxonomy.panelEyebrow")}
                  title={t("configuration.stages.taxonomy.panelTitle")}
                  intro={t("configuration.stages.taxonomy.panelIntro")}
                />
                <div
                  data-testid={DATA_TEST_ID.OPTION_LIST}
                  role="radiogroup"
                  aria-labelledby={DATA_TEST_ID.SECTION_LABEL}
                  className="flex flex-col gap-2"
                >
                  <span
                    id={DATA_TEST_ID.SECTION_LABEL}
                    data-testid={DATA_TEST_ID.SECTION_LABEL}
                    className="font-mono text-[11px] uppercase tracking-[0.08em] text-muted-2"
                  >
                    {t("configuration.stages.taxonomy.sectionLabel")}
                  </span>
                  {taxonomyModels.models.map((taxonomy) => (
                    <ModelOption
                      key={taxonomy.id}
                      modelId={taxonomy.id}
                      title={`${taxonomy.name} ${taxonomy.version}`}
                      description={taxonomy.description}
                      suffix={taxonomy.version}
                      recommended={
                        taxonomy.id === RECOMMENDED_TAXONOMY_MODEL_ID
                      }
                      selected={activeDraft.taxonomy_model_id === taxonomy.id}
                      onClick={() => handleTaxonomySelect(taxonomy.id)}
                    />
                  ))}
                </div>
              </>
            )}
          </section>
        </div>
      )}

      <SaveBar
        isDirty={configuration.isDirty}
        saveStatus={configuration.saveStatus}
        onSave={() => {
          void handleSave();
        }}
        onDiscard={configuration.discard}
      />

      <UnsavedChangesGuard
        open={guard.isPromptOpen}
        onConfirm={guard.confirm}
        onCancel={guard.cancel}
      />
    </div>
  );
}

interface PanelHeaderProps {
  eyebrow: string;
  title: string;
  intro: string;
}

function PanelHeader({ eyebrow, title, intro }: PanelHeaderProps) {
  return (
    <header className="flex flex-col gap-1.5">
      <span data-testid={DATA_TEST_ID.PANEL_EYEBROW} className="eyebrow">
        {eyebrow}
      </span>
      <h2 data-testid={DATA_TEST_ID.PANEL_TITLE} className="h-section m-0">
        {title}
      </h2>
      <p
        data-testid={DATA_TEST_ID.PANEL_INTRO}
        className="m-0 max-w-[640px] text-sm leading-relaxed text-muted"
      >
        {intro}
      </p>
    </header>
  );
}
