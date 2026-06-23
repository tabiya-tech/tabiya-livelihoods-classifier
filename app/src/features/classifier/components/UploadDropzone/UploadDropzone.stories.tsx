import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { UploadDropzone } from "./UploadDropzone";

const meta: Meta<typeof UploadDropzone> = {
  title: "Features/Classifier/UploadDropzone",
  component: UploadDropzone,
  parameters: { layout: "padded" },
  args: { onText: fn() },
};
export default meta;

type Story = StoryObj<typeof UploadDropzone>;

export const Default: Story = {};

export const Disabled: Story = {
  args: { disabled: true },
};
