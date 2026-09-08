import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { RevealOnceBanner } from "./RevealOnceBanner";

const meta: Meta<typeof RevealOnceBanner> = {
  title: "Features/Keys/RevealOnceBanner",
  component: RevealOnceBanner,
  parameters: { layout: "padded" },
  args: {
    apiKey: "AIzaSyDEMO0000000000000000000000000000000",
    onDismiss: fn(),
  },
};
export default meta;

type Story = StoryObj<typeof RevealOnceBanner>;

export const Default: Story = {};
