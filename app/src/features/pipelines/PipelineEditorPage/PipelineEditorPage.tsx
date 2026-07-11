/**
 * Pipeline editor page — compose the palette, canvas, drawer, and save bar
 * into a full-height editing surface.
 *
 * Mounted at:
 *   /pipelines/new          → create a brand-new pipeline
 *   /pipelines/:pipelineId  → edit an existing pipeline
 *
 * Layout:
 *   ┌──────────────────────────────────────────────────────────────┐
 *   │  Header: eyebrow + title                                     │
 *   ├────────────────────────┬─────────────────────────────────────┤
 *   │  PluginPalette (240px) │  PipelineCanvas (flex-1)            │
 *   └────────────────────────┴─────────────────────────────────────┘
 *   │  PipelineSaveBar (fixed bottom)                              │
 */

import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { ReactFlowProvider } from "reactflow";
import { useTranslation } from "react-i18next";
import { Spinner, useToast } from "@/components";
import type { PipelineStage } from "@/lib/api";
import { createPipeline, updatePipeline } from "@/lib/api";
import { routerPaths } from "@/routes/routerPaths";
import { UnsavedChangesGuard } from "@/features/configuration/components/UnsavedChangesGuard/UnsavedChangesGuard";
import { useUnsavedChangesGuard } from "@/features/configuration/hooks/useUnsavedChangesGuard";
import { InlineEditableTitle } from "../components/InlineEditableTitle/InlineEditableTitle";
import { PipelineCanvas } from "../components/PipelineCanvas/PipelineCanvas";
import { PipelineSaveBar } from "../components/PipelineSaveBar/PipelineSaveBar";
import { PluginPalette } from "../components/PluginPalette/PluginPalette";
import { StageDetailDrawer } from "../components/StageDetailDrawer/StageDetailDrawer";
import { usePipelineEditor } from "../hooks/usePipelineEditor";
import { useValidatePipeline } from "../hooks/useValidatePipeline";

const uniqueId = "c3d4e5f6-a7b8-4c9d-8e0f-1a2b3c4d5e6f";

export const DATA_TEST_ID = {
  CONTAINER: `pipeline-editor-page-container-${uniqueId}`,
  PALETTE_CONTAINER: `pipeline-editor-page-palette-container-${uniqueId}`,
  CANVAS_CONTAINER: `pipeline-editor-page-canvas-container-${uniqueId}`,
  DRAWER_CONTAINER: `pipeline-editor-page-drawer-container-${uniqueId}`,
  SAVE_BAR_CONTAINER: `pipeline-editor-page-save-bar-container-${uniqueId}`,
  LOADING: `pipeline-editor-page-loading-${uniqueId}`,
  ERROR: `pipeline-editor-page-error-${uniqueId}`,
  EMPTY_PROMPT: `pipeline-editor-page-empty-prompt-${uniqueId}`,
};

export function PipelineEditorPage() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const toast = useToast();
  const { pipelineId: pipelineIdParam } = useParams<{ pipelineId?: string }>();

  // `undefined` means new pipeline; actual string means editing existing.
  const existingPipelineId = pipelineIdParam;

  const editorState = usePipelineEditor({ pipelineId: existingPipelineId });

  const originalStages = editorState.pipeline?.stages ?? [];
  const originalName = editorState.pipeline?.name ?? "";

  const [stages, setStages] = useState<PipelineStage[]>([]);
  const [pipelineName, setPipelineName] = useState<string>("");
  const [isInitialized, setIsInitialized] = useState(false);

  // Initialize local state once loading completes. We use an effect here so
  // the initialization is deferred until after the first render that reports
  // "ready", avoiding the setState-during-render anti-pattern.
  useEffect(() => {
    if (editorState.status === "ready" && !isInitialized) {
      setStages(editorState.pipeline?.stages ?? []);
      setPipelineName(editorState.pipeline?.name ?? "");
      setIsInitialized(true);
    }
  }, [editorState.status, editorState.pipeline, isInitialized]);

  const [selectedStageIndex, setSelectedStageIndex] = useState<number | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const validationState = useValidatePipeline(stages);

  const isDirty =
    isInitialized &&
    (JSON.stringify(stages) !== JSON.stringify(originalStages) ||
      pipelineName !== originalName);

  // The app uses <BrowserRouter> (not a data router), so React Router's
  // useBlocker is unavailable. Use the app-wide navigation guard instead —
  // the same mechanism the Configuration page uses.
  const unsavedGuard = useUnsavedChangesGuard(isDirty && !isSaving);

  const unsavedGuardOpen = unsavedGuard.isPromptOpen;
  const handleProceedNavigation = unsavedGuard.confirm;
  const handleCancelNavigation = unsavedGuard.cancel;

  async function handleSave() {
    setIsSaving(true);
    try {
      if (existingPipelineId) {
        await updatePipeline(existingPipelineId, { name: pipelineName, stages });
      } else {
        await createPipeline({ name: pipelineName, stages });
      }
      navigate(routerPaths.PIPELINES);
    } catch {
      toast.show({
        message: t("pipelines.editor.toasts.saveError"),
        tone: "error",
      });
    } finally {
      setIsSaving(false);
    }
  }

  function handleCancel() {
    navigate(routerPaths.PIPELINES);
  }

  function handleConnectRejected() {
    toast.show({
      message: t("pipelines.editor.errors.slotMismatch"),
      tone: "error",
    });
  }

  function handleNodeSelect(stageIndex: number) {
    setSelectedStageIndex(stageIndex);
    setDrawerOpen(true);
  }

  function handleDrawerClose() {
    setDrawerOpen(false);
  }

  function handleStageChange(nextStage: PipelineStage) {
    if (selectedStageIndex === null) return;
    setStages((previousStages) =>
      previousStages.map((stage, index) =>
        index === selectedStageIndex ? nextStage : stage,
      ),
    );
  }

  function handleStageDelete() {
    if (selectedStageIndex === null) return;
    setStages((previousStages) =>
      previousStages.filter((_stage, index) => index !== selectedStageIndex),
    );
    setDrawerOpen(false);
    setSelectedStageIndex(null);
  }

  const isReadonly = editorState.pipeline?.is_readonly ?? false;
  const isNewPipeline = !existingPipelineId;
  const pageTitle = isNewPipeline
    ? t("pipelines.editor.title.new")
    : t("pipelines.editor.title.edit");

  const selectedStage =
    selectedStageIndex !== null ? stages[selectedStageIndex] : undefined;
  const selectedManifest =
    selectedStage ? editorState.manifests[selectedStage.plugin_id] : undefined;
  const selectedStageErrors =
    selectedStageIndex !== null
      ? validationState.issues.filter(
          (issue) => issue.stage_index === selectedStageIndex,
        )
      : [];

  // Build a synthetic pipeline object for the canvas (merging local stages).
  const canvasPipeline = editorState.pipeline
    ? { ...editorState.pipeline, stages }
    : {
        pipeline_id: "",
        user_id: "",
        name: pipelineName,
        stages,
        is_active: false,
        is_default: false,
        is_readonly: false,
        created_at: "",
        updated_at: "",
      };

  if (editorState.status === "loading") {
    return (
      <div
        data-testid={DATA_TEST_ID.CONTAINER}
        className="flex h-full items-center justify-center gap-2 text-sm text-muted"
      >
        <div data-testid={DATA_TEST_ID.LOADING} className="flex items-center gap-2">
          <Spinner />
          {t("common.loading")}
        </div>
      </div>
    );
  }

  if (editorState.status === "error") {
    return (
      <div
        data-testid={DATA_TEST_ID.CONTAINER}
        className="flex h-full items-center justify-center"
      >
        <div
          data-testid={DATA_TEST_ID.ERROR}
          className="text-sm text-error"
        >
          {t("pipelines.editor.toasts.loadError")}
        </div>
      </div>
    );
  }

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className="flex h-full flex-col overflow-hidden"
    >
      {/* Page header — title on the left, save toolbar on the right */}
      <header className="flex items-center justify-between gap-4 border-b border-line px-4 py-3 sm:px-8 sm:py-4">
        <div className="flex flex-col gap-1">
          <span className="eyebrow">{t("pipelines.editor.eyebrow")}</span>
          <InlineEditableTitle
            value={pipelineName}
            onChange={setPipelineName}
            placeholder={pageTitle}
            readOnly={isReadonly}
          />
        </div>
        <div data-testid={DATA_TEST_ID.SAVE_BAR_CONTAINER}>
          <PipelineSaveBar
            isReadonly={isReadonly}
            issues={validationState.issues}
            isValidating={validationState.status === "checking"}
            isDirty={isDirty}
            isSaving={isSaving}
            onSave={() => {
              void handleSave();
            }}
            onCancel={handleCancel}
          />
        </div>
      </header>

      {/* Main editing area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Plugin palette */}
        <aside
          data-testid={DATA_TEST_ID.PALETTE_CONTAINER}
          className="w-72 shrink-0 overflow-y-auto overflow-x-hidden border-r border-line bg-paper"
        >
          <PluginPalette />
        </aside>

        {/* Canvas area */}
        <main
          data-testid={DATA_TEST_ID.CANVAS_CONTAINER}
          className="relative flex flex-1 flex-col overflow-hidden"
        >
          {stages.length === 0 && isInitialized && (
            <div
              data-testid={DATA_TEST_ID.EMPTY_PROMPT}
              className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center"
            >
              <p className="max-w-xs text-center text-sm text-muted">
                {t("pipelines.editor.emptyPrompt")}
              </p>
            </div>
          )}

          <ReactFlowProvider>
            <PipelineCanvas
              pipeline={canvasPipeline}
              manifests={editorState.manifests}
              validationIssues={validationState.issues}
              editMode={!isReadonly}
              onStagesChange={setStages}
              onConnectRejected={handleConnectRejected}
              onNodeSelect={handleNodeSelect}
              pluginSummaries={editorState.pluginSummaries}
              className="h-full w-full"
            />
          </ReactFlowProvider>
        </main>
      </div>

      {/* Stage detail drawer */}
      <div data-testid={DATA_TEST_ID.DRAWER_CONTAINER}>
        <StageDetailDrawer
          open={drawerOpen}
          onClose={handleDrawerClose}
          stage={selectedStage}
          stageIndex={selectedStageIndex ?? undefined}
          manifest={selectedManifest}
          errors={selectedStageErrors}
          onChange={handleStageChange}
          onDelete={handleStageDelete}
        />
      </div>

      {/* Navigation guard */}
      <UnsavedChangesGuard
        open={unsavedGuardOpen}
        onConfirm={handleProceedNavigation}
        onCancel={handleCancelNavigation}
      />
    </div>
  );
}
