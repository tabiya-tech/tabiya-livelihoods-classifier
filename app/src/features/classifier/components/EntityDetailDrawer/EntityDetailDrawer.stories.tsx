import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { fixtureClassifyEntities } from "@/mocks/fixtures/classify";
import { EntityDetailDrawer } from "./EntityDetailDrawer";

const meta: Meta<typeof EntityDetailDrawer> = {
  title: "Features/Classifier/EntityDetailDrawer",
  component: EntityDetailDrawer,
  parameters: { layout: "fullscreen" },
  args: { open: true, onClose: fn() },
};
export default meta;

type Story = StoryObj<typeof EntityDetailDrawer>;

export const OccupationEntity: Story = {
  args: { entity: fixtureClassifyEntities[0] },
};

export const SkillEntity: Story = {
  args: { entity: fixtureClassifyEntities[1] },
};

export const QualificationEntity: Story = {
  args: { entity: fixtureClassifyEntities[4] },
};

export const EmptyMatches: Story = {
  args: {
    entity: { ...fixtureClassifyEntities[0], matches: [] },
  },
};

export const Closed: Story = {
  args: { open: false, entity: fixtureClassifyEntities[0] },
};
