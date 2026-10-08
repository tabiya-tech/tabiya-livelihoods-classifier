import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { fixtureClassifyEntities } from "@/mocks/fixtures/classify";
import { EntityTable } from "./EntityTable";

const meta: Meta<typeof EntityTable> = {
  title: "Features/Classifier/EntityTable",
  component: EntityTable,
  parameters: { layout: "padded" },
  args: { onEntityClick: fn(), entities: fixtureClassifyEntities },
};
export default meta;

type Story = StoryObj<typeof EntityTable>;

export const Default: Story = {};

export const WithSelection: Story = {
  args: { selectedEntityIndex: 1 },
};

export const Empty: Story = {
  args: { entities: [] },
};
