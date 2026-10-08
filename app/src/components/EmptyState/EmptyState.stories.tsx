import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { EmptyState } from "./EmptyState";
import { Button } from "@/components";

const meta: Meta<typeof EmptyState> = {
  title: "Primitives/EmptyState",
  component: EmptyState,
  parameters: { layout: "padded" },
};
export default meta;

type Story = StoryObj<typeof EmptyState>;

export const Default: Story = {
  args: {
    icon: "history",
    title: "No classifications yet",
    description: "Run your first classification to see it here.",
    action: (
      <Button variant="primary" onClick={fn()}>
        Open the Classifier
      </Button>
    ),
  },
};

export const Minimal: Story = {
  args: { title: "Nothing matches your filters" },
};
