import type { Meta, StoryObj } from "@storybook/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { ReactFlowProvider } from "reactflow";
import { ToastProvider } from "@/components";
import { NavigationGuardProvider } from "@/lib/navigationGuard";
import {
  fixtureDefaultTabiyaPipeline,
  fixtureRecruiterTuningPipeline,
} from "@/mocks/fixtures/pipelines";
import {
  seedPipelinesHandlersStore,
  resetPipelinesHandlersStore,
} from "@/mocks/handlers";
import { PipelineEditorPage } from "./PipelineEditorPage";

function EditorWrapper({ initialPath }: { initialPath: string }) {
  return (
    <MemoryRouter initialEntries={[initialPath]}>
      <NavigationGuardProvider>
        <ToastProvider>
          <ReactFlowProvider>
            <Routes>
              <Route path="/pipelines/new" element={<PipelineEditorPage />} />
              <Route path="/pipelines/:pipelineId" element={<PipelineEditorPage />} />
              <Route
                path="/pipelines"
                element={
                  <div style={{ padding: 32 }}>
                    ← Back at pipelines list
                  </div>
                }
              />
            </Routes>
          </ReactFlowProvider>
        </ToastProvider>
      </NavigationGuardProvider>
    </MemoryRouter>
  );
}

const meta: Meta<typeof EditorWrapper> = {
  title: "Features/Pipelines/PipelineEditorPage",
  component: EditorWrapper,
  parameters: { layout: "fullscreen" },
};
export default meta;

type Story = StoryObj<typeof EditorWrapper>;

export const NewPipeline: Story = {
  args: { initialPath: "/pipelines/new" },
  beforeEach() {
    resetPipelinesHandlersStore();
  },
};

export const ExistingEditable: Story = {
  args: {
    initialPath: `/pipelines/${fixtureRecruiterTuningPipeline.pipeline_id}`,
  },
  beforeEach() {
    seedPipelinesHandlersStore([fixtureRecruiterTuningPipeline]);
  },
};

export const ExistingReadonly: Story = {
  args: {
    initialPath: `/pipelines/${fixtureDefaultTabiyaPipeline.pipeline_id}`,
  },
  beforeEach() {
    seedPipelinesHandlersStore([fixtureDefaultTabiyaPipeline]);
  },
};
