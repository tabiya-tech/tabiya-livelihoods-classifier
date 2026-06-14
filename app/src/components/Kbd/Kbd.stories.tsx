import type { Meta, StoryObj } from "@storybook/react";
import { Kbd } from "./Kbd";

const meta: Meta<typeof Kbd> = {
  title: "Primitives/Kbd",
  component: Kbd,
  parameters: { layout: "centered" },
  args: { children: "⌘ K" },
};
export default meta;

type Story = StoryObj<typeof Kbd>;

export const Default: Story = {};

export const Combinations: Story = {
  render: () => (
    <div className="flex items-center gap-2">
      <Kbd>⌘ K</Kbd>
      <Kbd>⌘ ⇧ P</Kbd>
      <Kbd>Esc</Kbd>
      <Kbd>↵</Kbd>
    </div>
  ),
};
