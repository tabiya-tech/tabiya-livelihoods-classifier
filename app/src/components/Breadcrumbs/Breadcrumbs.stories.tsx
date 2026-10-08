import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { Breadcrumbs } from "@/components";

const meta: Meta<typeof Breadcrumbs> = {
  title: "Primitives/Breadcrumbs",
  component: Breadcrumbs,
  parameters: { layout: "padded" },
};
export default meta;

type Story = StoryObj<typeof Breadcrumbs>;

export const Default: Story = {
  args: {
    items: [
      { label: "Workspace", onClick: fn() },
      { label: "Classifier" },
    ],
  },
};

export const ThreeLevel: Story = {
  args: {
    items: [
      { label: "Documentation", onClick: fn() },
      { label: "Endpoints", onClick: fn() },
      { label: "POST /v1/classify" },
    ],
  },
};
