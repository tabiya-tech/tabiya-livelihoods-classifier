import type { Meta, StoryObj } from "@storybook/react";
import { ReactFlowProvider } from "reactflow";
import { fixtureNerManifest, fixtureNelManifest, fixtureTextInputManifest } from "@/mocks/fixtures/plugins";
import { StageNode } from "./StageNode";

const baseNodeProps = {
  id: "stage-0",
  type: "stageNode",
  selected: false,
  isConnectable: false,
  zIndex: 1,
  xPos: 0,
  yPos: 0,
  dragging: false,
};

const meta: Meta<typeof StageNode> = {
  title: "Features/Pipelines/StageNode",
  component: StageNode,
  parameters: { layout: "centered" },
  decorators: [
    function ReactFlowDecorator(Story) {
      return (
        <ReactFlowProvider>
          <Story />
        </ReactFlowProvider>
      );
    },
  ],
};
export default meta;

type Story = StoryObj<typeof StageNode>;

export const SourceStage: Story = {
  args: {
    ...baseNodeProps,
    data: {
      pluginId: fixtureTextInputManifest.plugin_id,
      manifest: fixtureTextInputManifest,
      status: "enabled",
      stageIndex: 0,
    },
  },
};

export const CoreStageWithConfig: Story = {
  args: {
    ...baseNodeProps,
    data: {
      pluginId: fixtureNerManifest.plugin_id,
      manifest: fixtureNerManifest,
      status: "enabled",
      stageIndex: 1,
      configPreview: "entity_types=occupation,skill",
    },
  },
};

export const DegradedStage: Story = {
  args: {
    ...baseNodeProps,
    data: {
      pluginId: fixtureNelManifest.plugin_id,
      manifest: fixtureNelManifest,
      status: "degraded",
      stageIndex: 2,
      configPreview: "top_k=5",
    },
  },
};

export const StageWithValidationError: Story = {
  args: {
    ...baseNodeProps,
    data: {
      pluginId: fixtureNelManifest.plugin_id,
      manifest: fixtureNelManifest,
      status: "enabled",
      stageIndex: 2,
      hasError: true,
    },
  },
};

export const NoManifest: Story = {
  args: {
    ...baseNodeProps,
    data: {
      pluginId: "tabiya.unknown.v99",
      stageIndex: 0,
    },
  },
};
