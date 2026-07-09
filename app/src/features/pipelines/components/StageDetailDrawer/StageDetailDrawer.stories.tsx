import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import {
  fixtureNerManifest,
  fixtureNelManifest,
  fixturePluginOptions,
} from "@/mocks/fixtures/plugins";
import type { PipelineStage, PipelineValidationIssue } from "@/lib/api";
import type { PluginOptionsState } from "../../hooks/usePluginOptions";
import { PluginOptionsOverrideContext } from "../../hooks/pipelinesOverrides";
import { StageDetailDrawer } from "./StageDetailDrawer";

const givenNerStage: PipelineStage = {
  plugin_id: fixtureNerManifest.plugin_id,
  config: {
    model_id: "tabiya/roberta-base-job-ner",
    entity_types: ["occupation", "skill"],
  },
};

const givenNelStage: PipelineStage = {
  plugin_id: fixtureNelManifest.plugin_id,
  config: {
    nel_model_id: "all-MiniLM-L6-v2",
    taxonomy_model_id: "esco-v1.1",
    top_k: 5,
    min_similarity: 0,
  },
};

const meta: Meta<typeof StageDetailDrawer> = {
  title: "Features/Pipelines/StageDetailDrawer",
  component: StageDetailDrawer,
  parameters: { layout: "fullscreen" },
  args: {
    open: true,
    onClose: fn(),
    onChange: fn(),
    onDelete: fn(),
  },
  decorators: [
    function PluginOptionsDecorator(Story) {
      const readyState: PluginOptionsState = {
        status: "ready",
        options: fixturePluginOptions["tabiya.ner.v1"].model_id.options,
        error: null,
      };
      return (
        <PluginOptionsOverrideContext.Provider value={readyState}>
          <Story />
        </PluginOptionsOverrideContext.Provider>
      );
    },
  ],
};
export default meta;

type Story = StoryObj<typeof StageDetailDrawer>;

export const Open: Story = {
  args: {
    stage: givenNerStage,
    stageIndex: 1,
    manifest: fixtureNerManifest,
  },
};

export const WithErrors: Story = {
  args: {
    stage: givenNelStage,
    stageIndex: 2,
    manifest: fixtureNelManifest,
    errors: [
      {
        code: "MISSING_REQUIRED",
        message: "nel_model_id is required",
        stage_index: 2,
        plugin_id: fixtureNelManifest.plugin_id,
      } satisfies PipelineValidationIssue,
      {
        code: "INVALID_VALUE",
        message: "top_k must be between 1 and 50",
        stage_index: 2,
        plugin_id: fixtureNelManifest.plugin_id,
      } satisfies PipelineValidationIssue,
    ],
  },
};

export const Readonly: Story = {
  name: "No stage selected",
  args: {
    stage: undefined,
    stageIndex: undefined,
    manifest: undefined,
  },
};
