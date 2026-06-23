import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { fixtureClassifyEntities } from "@/mocks/fixtures/classify";
import { EntityRow } from "./EntityRow";

const meta: Meta<typeof EntityRow> = {
  title: "Features/Classifier/EntityRow",
  component: EntityRow,
  parameters: { layout: "padded" },
  args: { onClick: fn(), entityIndex: 0 },
};
export default meta;

type Story = StoryObj<typeof EntityRow>;

export const OccupationRow: Story = {
  args: { entity: fixtureClassifyEntities[0] },
};

export const SkillRow: Story = {
  args: { entity: fixtureClassifyEntities[1], entityIndex: 1 },
};

export const QualificationRow: Story = {
  args: { entity: fixtureClassifyEntities[4], entityIndex: 4 },
};

export const Selected: Story = {
  args: { entity: fixtureClassifyEntities[0], isSelected: true },
};

export const EmptyMatches: Story = {
  args: { entity: { ...fixtureClassifyEntities[0], matches: [] } },
};
