import type { Meta, StoryObj } from "@storybook/react";
import { Tag } from "./Tag";

const meta: Meta<typeof Tag> = {
  title: "Primitives/Tag",
  component: Tag,
  parameters: { layout: "centered" },
  argTypes: {
    tone: { control: "select", options: ["neutral", "lime", "teal", "muted", "danger"] },
    size: { control: "select", options: ["sm", "md"] },
  },
  args: { children: "recommended" },
};
export default meta;

type Story = StoryObj<typeof Tag>;

export const Default: Story = {};
export const Lime: Story = { args: { tone: "lime", children: "connected" } };
export const WithDot: Story = { args: { tone: "teal", dot: true, children: "active" } };
export const Small: Story = { args: { size: "sm", children: "125M params" } };

export const Tones: Story = {
  render: () => (
    <div className="flex flex-wrap items-center gap-2">
      <Tag tone="neutral">neutral</Tag>
      <Tag tone="lime">connected</Tag>
      <Tag tone="teal" dot>active</Tag>
      <Tag tone="muted">not connected</Tag>
      <Tag tone="danger">revoked</Tag>
    </div>
  ),
};
