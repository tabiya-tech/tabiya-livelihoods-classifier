import { useState } from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { MemoryRouter } from "react-router-dom";
import { ToastProvider } from "@/components";
import { fixtureClassifyResponse } from "@/mocks/fixtures/classify";
import {
  fixtureDefaultTabiyaPipeline,
  fixtureRecruiterTuningPipeline,
} from "@/mocks/fixtures/pipelines";
import type { ClassifyRequest, ClassifyResponse } from "@/lib/api";
import { ClassifierOverridesProvider } from "../hooks/classifierOverrides";
import { ClassifierPipelineOverridesProvider } from "../hooks/classifierPipelineOverrides";
import type {
  ClassifyState,
  ClassifyStatus,
} from "../hooks/useClassify";
import type { ActivePipelineSnapshot } from "../hooks/useActivePipeline";
import { ClassifierPage } from "./ClassifierPage";

const readyActivePipelineSnapshot: ActivePipelineSnapshot = {
  status: "ready",
  pipelines: [fixtureDefaultTabiyaPipeline, fixtureRecruiterTuningPipeline],
  activePipeline: fixtureDefaultTabiyaPipeline,
  error: null,
  setActivePipeline: async () => undefined,
};

interface HarnessProps {
  initialStatus?: ClassifyStatus;
  initialResponse?: ClassifyResponse | null;
  runImpl?: (payload: ClassifyRequest) => Promise<ClassifyResponse>;
  children: React.ReactNode;
}

/**
 * Stateful harness wrapping the page so clicking Run + Clear flips state
 * without ever touching the network.
 */
function ClassifierHarness({
  initialStatus = "idle",
  initialResponse = null,
  runImpl,
  children,
}: HarnessProps) {
  const [status, setStatus] = useState<ClassifyStatus>(initialStatus);
  const [response, setResponse] = useState<ClassifyResponse | null>(
    initialResponse,
  );

  const value: ClassifyState = {
    status,
    response,
    error: null,
    run: async (payload) => {
      setStatus("running");
      try {
        const next = runImpl
          ? await runImpl(payload)
          : fixtureClassifyResponse;
        setResponse(next);
        setStatus("done");
        return next;
      } catch (caught) {
        setStatus("error");
        throw caught;
      }
    },
    reset: () => {
      setStatus("idle");
      setResponse(null);
    },
  };

  return (
    <ClassifierPipelineOverridesProvider
      activePipeline={readyActivePipelineSnapshot}
    >
      <ClassifierOverridesProvider classify={value}>
        {children}
      </ClassifierOverridesProvider>
    </ClassifierPipelineOverridesProvider>
  );
}

const meta: Meta<typeof ClassifierPage> = {
  title: "Features/Classifier/ClassifierPage",
  component: ClassifierPage,
  parameters: { layout: "fullscreen" },
  decorators: [
    function StoryWithShell(StoryComponent) {
      return (
        <MemoryRouter initialEntries={["/classifier"]}>
          <ToastProvider>
            <StoryComponent />
          </ToastProvider>
        </MemoryRouter>
      );
    },
  ],
};
export default meta;

type Story = StoryObj<typeof ClassifierPage>;

export const Empty: Story = {
  render: function EmptyStory() {
    return (
      <ClassifierHarness>
        <ClassifierPage />
      </ClassifierHarness>
    );
  },
};

export const RunningRequest: Story = {
  render: function RunningStory() {
    return (
      <ClassifierHarness initialStatus="running">
        <ClassifierPage />
      </ClassifierHarness>
    );
  },
};

export const WithResults: Story = {
  render: function WithResultsStory() {
    return (
      <ClassifierHarness
        initialStatus="done"
        initialResponse={fixtureClassifyResponse}
      >
        <ClassifierPage />
      </ClassifierHarness>
    );
  },
};

export const RunFails: Story = {
  render: function RunFailsStory() {
    const onRun = fn(async () => {
      throw new Error("Backend unreachable");
    });
    return (
      <ClassifierHarness runImpl={onRun as never}>
        <ClassifierPage />
      </ClassifierHarness>
    );
  },
};
