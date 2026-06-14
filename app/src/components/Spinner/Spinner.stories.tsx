import type { Meta, StoryObj } from "@storybook/react";
import { Spinner } from "./Spinner";

const meta: Meta<typeof Spinner> = {
  title: "Primitives/Spinner",
  component: Spinner,
  parameters: { layout: "centered" },
};
export default meta;

type Story = StoryObj<typeof Spinner>;

export const Default: Story = { args: { size: 16 } };
export const Large: Story = { args: { size: 32 } };

export const Sizes: Story = {
  render: () => (
    <div className="flex items-center gap-4 text-navy">
      <Spinner size={12} />
      <Spinner size={16} />
      <Spinner size={24} />
      <Spinner size={32} />
    </div>
  ),
};
