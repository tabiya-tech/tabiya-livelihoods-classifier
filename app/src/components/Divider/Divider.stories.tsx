import type { Meta, StoryObj } from "@storybook/react";
import { Divider } from "./Divider";

const meta: Meta<typeof Divider> = {
  title: "Primitives/Divider",
  component: Divider,
  parameters: { layout: "padded" },
};
export default meta;

type Story = StoryObj<typeof Divider>;

export const Solid: Story = {
  render: () => (
    <div className="max-w-md">
      <p className="text-muted">Above</p>
      <Divider />
      <p className="text-muted">Below</p>
    </div>
  ),
};

export const Dashed: Story = {
  render: () => (
    <div className="max-w-md">
      <p className="text-muted">Above</p>
      <Divider dashed />
      <p className="text-muted">Below</p>
    </div>
  ),
};
