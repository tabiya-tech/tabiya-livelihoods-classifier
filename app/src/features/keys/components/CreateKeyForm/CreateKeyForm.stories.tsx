import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { CreateKeyForm } from "./CreateKeyForm";

const meta: Meta<typeof CreateKeyForm> = {
  title: "Features/Keys/CreateKeyForm",
  component: CreateKeyForm,
  parameters: { layout: "padded" },
  args: { onSubmit: fn() },
};
export default meta;

type Story = StoryObj<typeof CreateKeyForm>;

export const Default: Story = {};

export const Submitting: Story = {
  args: { isSubmitting: true },
};

export const MaxReached: Story = {
  args: { maxReached: true, maxKeys: 5 },
};
