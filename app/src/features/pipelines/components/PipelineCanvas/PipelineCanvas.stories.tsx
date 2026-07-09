import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { ReactFlowProvider } from "reactflow";
import {
  fixtureDefaultTabiyaPipeline,
  fixtureRecruiterTuningPipeline,
} from "@/mocks/fixtures/pipelines";
import {
  fixturePluginManifests,
  fixtureTextInputManifest,
  fixtureNelManifest,
  fixtureResultsManifest,
} from "@/mocks/fixtures/plugins";
import type { Pipeline } from "@/lib/api";
import { PipelineCanvas } from "./PipelineCanvas";

const meta: Meta<typeof PipelineCanvas> = {
  title: "Features/Pipelines/PipelineCanvas",
  component: PipelineCanvas,
  parameters: { layout: "fullscreen" },
  decorators: [
    function ReactFlowDecorator(Story) {
      return (
        <ReactFlowProvider>
          <div style={{ height: "500px" }}>
            <Story />
          </div>
        </ReactFlowProvider>
      );
    },
  ],
};
export default meta;

type Story = StoryObj<typeof PipelineCanvas>;

export const DefaultTabiyaPipeline: Story = {
  args: {
    pipeline: fixtureDefaultTabiyaPipeline,
    manifests: fixturePluginManifests,
  },
};

export const RecruiterTuningPipeline: Story = {
  args: {
    pipeline: fixtureRecruiterTuningPipeline,
    manifests: fixturePluginManifests,
  },
};

const invalidChainPipeline: Pipeline = {
  pipeline_id: "pipeline-invalid",
  user_id: "local-user",
  name: "Invalid Chain",
  stages: [
    { plugin_id: fixtureTextInputManifest.plugin_id, config: {} },
    { plugin_id: fixtureNelManifest.plugin_id, config: {} },
    { plugin_id: fixtureResultsManifest.plugin_id, config: {} },
  ],
  is_active: false,
  is_default: false,
  is_readonly: false,
  created_at: "2026-07-09T00:00:00.000Z",
  updated_at: "2026-07-09T00:00:00.000Z",
};

export const InvalidChainWithMismatch: Story = {
  args: {
    pipeline: invalidChainPipeline,
    manifests: {
      [fixtureTextInputManifest.plugin_id]: fixtureTextInputManifest,
      [fixtureNelManifest.plugin_id]: fixtureNelManifest,
      [fixtureResultsManifest.plugin_id]: fixtureResultsManifest,
    },
    validationIssues: [
      {
        code: "slot_mismatch",
        message: "Output slot RawText does not match input slot Entities",
        stage_index: 1,
        plugin_id: fixtureNelManifest.plugin_id,
      },
    ],
  },
};

export const EditMode: Story = {
  args: {
    pipeline: fixtureRecruiterTuningPipeline,
    manifests: fixturePluginManifests,
    editMode: true,
    onStagesChange: fn(),
    onConnectRejected: fn(),
    onNodeSelect: fn(),
  },
};
