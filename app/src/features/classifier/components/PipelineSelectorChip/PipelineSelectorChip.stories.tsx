import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { MemoryRouter } from "react-router-dom";
import type { Pipeline } from "@/lib/api";
import { PipelineSelectorChip } from "./PipelineSelectorChip";

const fixtureDefaultPipeline: Pipeline = {
  pipeline_id: "pipeline-default",
  user_id: "local-user",
  name: "Default Tabiya",
  stages: [],
  is_active: true,
  is_default: true,
  is_readonly: true,
  created_at: "2026-01-01T00:00:00.000Z",
  updated_at: "2026-01-01T00:00:00.000Z",
};

const fixtureRecruiterPipeline: Pipeline = {
  pipeline_id: "pipeline-recruiter",
  user_id: "local-user",
  name: "Recruiter tuning",
  stages: [],
  is_active: false,
  is_default: false,
  is_readonly: false,
  created_at: "2026-02-01T00:00:00.000Z",
  updated_at: "2026-02-01T00:00:00.000Z",
};

const meta: Meta<typeof PipelineSelectorChip> = {
  title: "Features/Classifier/PipelineSelectorChip",
  component: PipelineSelectorChip,
  parameters: { layout: "padded" },
  decorators: [
    function StoryWithRouter(StoryComponent) {
      return (
        <MemoryRouter>
          <StoryComponent />
        </MemoryRouter>
      );
    },
  ],
  args: {
    onPipelineChange: fn(),
  },
};
export default meta;

type Story = StoryObj<typeof PipelineSelectorChip>;

export const Idle: Story = {
  args: {
    pipelines: [fixtureDefaultPipeline, fixtureRecruiterPipeline],
    selectedPipelineId: fixtureDefaultPipeline.pipeline_id,
    isLoading: false,
  },
};

export const Loading: Story = {
  args: {
    pipelines: [],
    selectedPipelineId: null,
    isLoading: true,
  },
};

export const Open: Story = {
  args: {
    pipelines: [fixtureDefaultPipeline, fixtureRecruiterPipeline],
    selectedPipelineId: fixtureDefaultPipeline.pipeline_id,
    isLoading: false,
    defaultOpen: true,
  },
};
