import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { RevokeConfirmModal } from "./RevokeConfirmModal";

const meta: Meta<typeof RevokeConfirmModal> = {
  title: "Features/Keys/RevokeConfirmModal",
  component: RevokeConfirmModal,
  parameters: { layout: "padded" },
  args: {
    keyLabel: "analyst-laptop",
    onConfirm: fn(),
    onCancel: fn(),
  },
};
export default meta;

type Story = StoryObj<typeof RevokeConfirmModal>;

export const Open: Story = {
  args: { open: true },
};

export const Submitting: Story = {
  args: { open: true, isSubmitting: true },
};

export const Closed: Story = {
  args: { open: false },
};
