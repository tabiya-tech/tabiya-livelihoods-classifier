import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { Input } from "./Input";

const meta: Meta<typeof Input> = {
  title: "Primitives/Input",
  component: Input,
  parameters: { layout: "padded" },
  args: { placeholder: "you@tabiya.org", onChange: fn(), onFocus: fn(), onBlur: fn() },
};
export default meta;

type Story = StoryObj<typeof Input>;

export const Default: Story = {};
export const Mono: Story = { args: { mono: true, placeholder: "tabiya_sk_…" } };
export const Invalid: Story = { args: { invalid: true, defaultValue: "not-an-email" } };
export const Password: Story = { args: { type: "password", defaultValue: "secret" } };
