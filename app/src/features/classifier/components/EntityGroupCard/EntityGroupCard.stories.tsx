import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { fixtureClassifyEntities } from "@/mocks/fixtures/classify";
import { EntityGroupCard } from "./EntityGroupCard";

const occupations = fixtureClassifyEntities
  .map((entity, entityIndex) => ({ entity, entityIndex }))
  .filter(({ entity }) => entity.entity_type === "occupation");

const skills = fixtureClassifyEntities
  .map((entity, entityIndex) => ({ entity, entityIndex }))
  .filter(({ entity }) => entity.entity_type === "skill");

const qualifications = fixtureClassifyEntities
  .map((entity, entityIndex) => ({ entity, entityIndex }))
  .filter(({ entity }) => entity.entity_type === "qualification");

const meta: Meta<typeof EntityGroupCard> = {
  title: "Features/Classifier/EntityGroupCard",
  component: EntityGroupCard,
  parameters: { layout: "padded" },
  args: { onEntityClick: fn() },
};
export default meta;

type Story = StoryObj<typeof EntityGroupCard>;

export const Skills: Story = {
  args: { entityType: "skill", entries: skills },
};

export const Occupations: Story = {
  args: { entityType: "occupation", entries: occupations },
};

export const Qualifications: Story = {
  args: { entityType: "qualification", entries: qualifications },
};

export const WithSelection: Story = {
  args: {
    entityType: "skill",
    entries: skills,
    selectedEntityIndex: skills[1]?.entityIndex,
  },
};
