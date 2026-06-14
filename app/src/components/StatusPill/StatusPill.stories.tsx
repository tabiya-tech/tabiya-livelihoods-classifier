import type { Meta, StoryObj } from "@storybook/react";
import { StatusPill } from "./StatusPill";

const meta: Meta<typeof StatusPill> = {
  title: "Primitives/StatusPill",
  component: StatusPill,
  parameters: { layout: "centered" },
  argTypes: {
    status: { control: "select", options: ["healthy", "degraded", "down", "unknown"] },
  },
  args: { children: "API healthy · v1.0.0" },
};
export default meta;

type Story = StoryObj<typeof StatusPill>;

export const Healthy: Story = { args: { status: "healthy" } };
export const Degraded: Story = { args: { status: "degraded", children: "API degraded" } };
export const Down: Story = { args: { status: "down", children: "API offline" } };
export const Unknown: Story = { args: { status: "unknown", children: "API checking…" } };
