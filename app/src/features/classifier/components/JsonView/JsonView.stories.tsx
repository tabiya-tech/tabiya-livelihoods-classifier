import type { Meta, StoryObj } from "@storybook/react";
import { fixtureClassifyResponse } from "@/mocks/fixtures/classify";
import { JsonView } from "./JsonView";

const meta: Meta<typeof JsonView> = {
  title: "Features/Classifier/JsonView",
  component: JsonView,
  parameters: { layout: "padded" },
};
export default meta;

type Story = StoryObj<typeof JsonView>;

export const Default: Story = {
  args: { value: fixtureClassifyResponse },
};

export const EmptyResponse: Story = {
  args: { value: { entities: [], metadata: {} } },
};
