/**
 * Tabiya Pipelines page — list / activate / clone / delete against /v2/pipelines.
 *
 * Composition:
 *   - Page header (eyebrow + title + intro).
 *   - {@link PipelinesTable} renders the pipeline list.
 *   - Loading, empty, and error states handled inline.
 *   - Error toasts for activate / clone / delete / load failures.
 */

import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router-dom";
import { Button, EmptyState, Spinner, useToast } from "@/components";
import { routerPaths } from "@/routes/routerPaths";
import { PipelinesTable } from "../components/PipelinesTable/PipelinesTable";
import { useActivatePipeline } from "../hooks/useActivatePipeline";
import { useClonePipeline } from "../hooks/useClonePipeline";
import { useDeletePipeline } from "../hooks/useDeletePipeline";
import { usePipelinesList } from "../hooks/usePipelinesList";

const uniqueId = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d";

export const DATA_TEST_ID = {
  CONTAINER: `pipelines-page-container-${uniqueId}`,
  EYEBROW: `pipelines-page-eyebrow-${uniqueId}`,
  TITLE: `pipelines-page-title-${uniqueId}`,
  INTRO: `pipelines-page-intro-${uniqueId}`,
  LOADING: `pipelines-page-loading-${uniqueId}`,
  LOAD_ERROR: `pipelines-page-load-error-${uniqueId}`,
  EMPTY_STATE: `pipelines-page-empty-state-${uniqueId}`,
};

export function PipelinesPage() {
  const { t } = useTranslation();
  const toast = useToast();
  const navigate = useNavigate();

  const pipelinesList = usePipelinesList();
  const activatePipeline = useActivatePipeline({
    onSuccess: () => pipelinesList.refetch(),
  });
  const clonePipeline = useClonePipeline({
    onSuccess: () => pipelinesList.refetch(),
  });
  const deletePipeline = useDeletePipeline({
    onSuccess: () => pipelinesList.refetch(),
  });

  const pendingId =
    activatePipeline.pendingId ??
    clonePipeline.pendingId ??
    deletePipeline.pendingId;

  async function handleActivate(pipelineId: string) {
    try {
      await activatePipeline.activate(pipelineId);
    } catch {
      toast.show({
        message: t("pipelines.list.toasts.activateError"),
        tone: "error",
      });
    }
  }

  async function handleClone(pipelineId: string) {
    try {
      await clonePipeline.clone(pipelineId);
    } catch {
      toast.show({
        message: t("pipelines.list.toasts.cloneError"),
        tone: "error",
      });
    }
  }

  async function handleDelete(pipelineId: string) {
    try {
      await deletePipeline.delete(pipelineId);
    } catch {
      toast.show({
        message: t("pipelines.list.toasts.deleteError"),
        tone: "error",
      });
    }
  }

  const isLoading = pipelinesList.status === "loading";
  const loadError = pipelinesList.error;
  const pipelines = pipelinesList.pipelines;

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className="mx-auto flex w-full max-w-[1080px] flex-col gap-8 px-8 py-10"
    >
      <header className="flex flex-col gap-2">
        <span data-testid={DATA_TEST_ID.EYEBROW} className="eyebrow">
          {t("pipelines.list.eyebrow")}
        </span>
        <div className="flex items-start justify-between gap-4">
          <h1 data-testid={DATA_TEST_ID.TITLE} className="h-page m-0">
            {t("pipelines.list.title")}
          </h1>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              onClick={() => navigate(routerPaths.PIPELINE_LIBRARY)}
            >
              {t("pipelines.list.libraryButton")}
            </Button>
            <Button
              variant="primary"
              onClick={() => navigate(routerPaths.PIPELINE_NEW)}
            >
              {t("pipelines.list.newButton")}
            </Button>
          </div>
        </div>
        <p
          data-testid={DATA_TEST_ID.INTRO}
          className="m-0 max-w-[680px] text-sm leading-relaxed text-muted"
        >
          {t("pipelines.list.intro")}
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
          <EmptyState icon="close" title={t("pipelines.list.toasts.loadError")} />
        </div>
      )}

      {!isLoading && !loadError && pipelines.length === 0 && (
        <div data-testid={DATA_TEST_ID.EMPTY_STATE}>
          <EmptyState
            icon="pipelines"
            title={t("pipelines.list.empty.title")}
            description={t("pipelines.list.empty.description")}
          />
        </div>
      )}

      {!isLoading && !loadError && pipelines.length > 0 && (
        <PipelinesTable
          pipelines={pipelines}
          pendingId={pendingId}
          onActivate={(pipelineId) => {
            void handleActivate(pipelineId);
          }}
          onEdit={(pipelineId) => {
            navigate(`/pipelines/${pipelineId}`);
          }}
          onClone={(pipelineId) => {
            void handleClone(pipelineId);
          }}
          onDelete={(pipelineId) => {
            void handleDelete(pipelineId);
          }}
        />
      )}
    </div>
  );
}
