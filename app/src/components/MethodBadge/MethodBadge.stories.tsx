import type { Meta, StoryObj } from "@storybook/react";
import { MethodBadge } from "./MethodBadge";

const meta: Meta<typeof MethodBadge> = {
  title: "Primitives/MethodBadge",
  component: MethodBadge,
  parameters: { layout: "centered" },
  argTypes: {
    method: { control: "select", options: ["GET", "POST", "PUT", "DELETE"] },
  },
  args: { method: "POST" },
};
export default meta;

type Story = StoryObj<typeof MethodBadge>;

export const Default: Story = {};

export const AllMethods: Story = {
  render: () => (
    <div className="flex items-center gap-2">
      <MethodBadge method="GET" />
      <MethodBadge method="POST" />
      <MethodBadge method="PUT" />
      <MethodBadge method="DELETE" />
    </div>
  ),
};
