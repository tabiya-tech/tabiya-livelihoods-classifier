import { useState, type ReactNode } from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { BrowserRouter } from "react-router-dom";
import { ToastProvider } from "@/components";
import type { Pipeline } from "@/lib/api";
import {
  fixturePipelines,
  fixtureDefaultTabiyaPipeline,
  fixtureRecruiterTuningPipeline,
} from "@/mocks/fixtures/pipelines";
import { PipelinesOverridesProvider } from "../hooks/pipelinesOverrides";
import type { PipelinesListSnapshot } from "../hooks/usePipelinesList";
import type { ActivatePipelineState } from "../hooks/useActivatePipeline";
import type { ClonePipelineState } from "../hooks/useClonePipeline";
import type { DeletePipelineState } from "../hooks/useDeletePipeline";
import { PipelinesPage } from "./PipelinesPage";

interface PipelinesHarnessProps {
  seedPipelines: Pipeline[];
  listStatus?: PipelinesListSnapshot["status"];
  children: ReactNode;
}

function PipelinesHarness({
  seedPipelines,
  listStatus = "ready",
  children,
}: PipelinesHarnessProps) {
  const [pipelines, setPipelines] = useState<Pipeline[]>(seedPipelines);
  const [activatePendingId, setActivatePendingId] = useState<string | null>(
    null,
  );
  const [clonePendingId, setClonePendingId] = useState<string | null>(null);
  const [deletePendingId, setDeletePendingId] = useState<string | null>(null);

  const listError: Error | null =
    listStatus === "error" ? new Error("Story error") : null;

  const pipelinesListOverride: PipelinesListSnapshot = {
    status: listStatus,
    pipelines,
    error: listError,
    refetch: async () => undefined,
  };

  const activatePipelineOverride: ActivatePipelineState = {
    status: activatePendingId ? "submitting" : "idle",
    error: null,
    pendingId: activatePendingId,
    activate: async (pipelineId: string) => {
      setActivatePendingId(pipelineId);
      const updated = pipelines.map((pipeline) =>
        pipeline.pipeline_id === pipelineId
          ? { ...pipeline, is_active: true }
          : { ...pipeline, is_active: false },
      );
      setPipelines(updated);
      setActivatePendingId(null);
      return updated.find((pipeline) => pipeline.pipeline_id === pipelineId)!;
    },
  };

  const clonePipelineOverride: ClonePipelineState = {
    status: clonePendingId ? "submitting" : "idle",
    error: null,
    pendingId: clonePendingId,
    clone: async (pipelineId: string) => {
      setClonePendingId(pipelineId);
      const source = pipelines.find(
        (pipeline) => pipeline.pipeline_id === pipelineId,
      )!;
      const cloned: Pipeline = {
        ...source,
        pipeline_id: `${pipelineId}-clone-${Date.now()}`,
        name: `${source.name} (copy)`,
        is_active: false,
        is_readonly: false,
      };
      setPipelines((previous) => [...previous, cloned]);
      setClonePendingId(null);
      return cloned;
    },
  };

  const deletePipelineOverride: DeletePipelineState = {
    status: deletePendingId ? "submitting" : "idle",
    error: null,
    pendingId: deletePendingId,
    delete: async (pipelineId: string) => {
      setDeletePendingId(pipelineId);
      setPipelines((previous) =>
        previous.filter((pipeline) => pipeline.pipeline_id !== pipelineId),
      );
      setDeletePendingId(null);
    },
  };

  return (
    <PipelinesOverridesProvider
      pipelinesList={pipelinesListOverride}
      activatePipeline={activatePipelineOverride}
      clonePipeline={clonePipelineOverride}
      deletePipeline={deletePipelineOverride}
    >
      {children}
    </PipelinesOverridesProvider>
  );
}

interface PageWrapperProps {
  seedPipelines: Pipeline[];
  listStatus?: PipelinesListSnapshot["status"];
}

function PageWrapper({ seedPipelines, listStatus }: PageWrapperProps) {
  return (
    <PipelinesHarness seedPipelines={seedPipelines} listStatus={listStatus}>
      <PipelinesPage />
    </PipelinesHarness>
  );
}

const meta: Meta<typeof PageWrapper> = {
  title: "Features/Pipelines/PipelinesPage",
  component: PageWrapper,
  parameters: { layout: "fullscreen" },
  decorators: [
    function StoryWithProviders(StoryComponent) {
      return (
        <BrowserRouter>
          <ToastProvider>
            <StoryComponent />
          </ToastProvider>
        </BrowserRouter>
      );
    },
  ],
};
export default meta;

type Story = StoryObj<typeof PageWrapper>;

export const Default: Story = {
  args: { seedPipelines: fixturePipelines },
};

export const Loading: Story = {
  args: { seedPipelines: [], listStatus: "loading" },
};

export const Empty: Story = {
  args: { seedPipelines: [] },
};

export const LoadError: Story = {
  args: { seedPipelines: [], listStatus: "error" },
};

export const ManyPipelines: Story = {
  args: {
    seedPipelines: [
      fixtureDefaultTabiyaPipeline,
      fixtureRecruiterTuningPipeline,
      ...Array.from({ length: 6 }, (_unused, index) => ({
        pipeline_id: `pipeline-extra-${index}`,
        user_id: "local-user",
        name: `Custom pipeline ${index + 1}`,
        stages: [],
        is_active: false,
        is_default: false,
        is_readonly: false,
        created_at: "2026-03-01T00:00:00.000Z",
        updated_at: "2026-07-01T00:00:00.000Z",
      })),
    ],
  },
};
