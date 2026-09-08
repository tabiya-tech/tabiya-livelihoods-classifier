import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { fixtureApiKeyMetadata } from "@/mocks/fixtures/apiKeys";
import { KeyTable } from "./KeyTable";

const meta: Meta<typeof KeyTable> = {
  title: "Features/Keys/KeyTable",
  component: KeyTable,
  parameters: { layout: "padded" },
  args: { onRevoke: fn(), keys: fixtureApiKeyMetadata },
};
export default meta;

type Story = StoryObj<typeof KeyTable>;

export const Default: Story = {};

export const Revoking: Story = {
  args: { pendingKeyId: fixtureApiKeyMetadata[0].key_id },
};
