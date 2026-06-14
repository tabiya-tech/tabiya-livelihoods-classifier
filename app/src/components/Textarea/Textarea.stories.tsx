import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { Textarea } from "./Textarea";

const meta: Meta<typeof Textarea> = {
  title: "Primitives/Textarea",
  component: Textarea,
  parameters: { layout: "padded" },
  args: {
    rows: 5,
    placeholder: "Paste a job description here…",
    onChange: fn(),
    onFocus: fn(),
    onBlur: fn(),
  },
};
export default meta;

type Story = StoryObj<typeof Textarea>;

export const Default: Story = {};
export const Mono: Story = { args: { mono: true } };
export const Invalid: Story = { args: { invalid: true, defaultValue: "missing required terms" } };
