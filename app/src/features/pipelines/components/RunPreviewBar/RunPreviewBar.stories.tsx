import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { RunPreviewBar } from "./RunPreviewBar";

const meta: Meta<typeof RunPreviewBar> = {
  title: "Features/Pipelines/RunPreviewBar",
  component: RunPreviewBar,
  parameters: { layout: "centered" },
  args: {
    onActiveStageChange: fn(),
  },
};
export default meta;

type Story = StoryObj<typeof RunPreviewBar>;

export const Idle: Story = {
  args: {
    stageCount: 3,
  },
};

export const Playing: Story = {
  name: "Playing (slow)",
  args: {
    stageCount: 4,
    // Slow the animation down so it's observable inside the Storybook frame.
    stepIntervalMs: 1200,
  },
};
