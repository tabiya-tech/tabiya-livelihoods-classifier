import type { Meta, StoryObj } from "@storybook/react";
import { Eyebrow } from "./Eyebrow";

const meta: Meta<typeof Eyebrow> = {
  title: "Primitives/Eyebrow",
  component: Eyebrow,
  parameters: { layout: "padded" },
  args: { children: "Workspace · Classifier" },
};
export default meta;

type Story = StoryObj<typeof Eyebrow>;

export const Default: Story = {};

export const AboveHeading: Story = {
  render: () => (
    <div>
      <Eyebrow>Pipeline · Stage 01</Eyebrow>
      <h1 className="h-page mt-2">Named Entity Recognition</h1>
    </div>
  ),
};
