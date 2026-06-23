import type { Meta, StoryObj } from "@storybook/react";
import { EntitySwatch } from "./EntitySwatch";

const meta: Meta<typeof EntitySwatch> = {
  title: "Features/Classifier/EntitySwatch",
  component: EntitySwatch,
  parameters: { layout: "centered" },
  args: { entityType: "occupation" },
};
export default meta;

type Story = StoryObj<typeof EntitySwatch>;

export const Occupation: Story = {};

export const Skill: Story = { args: { entityType: "skill" } };

export const Qualification: Story = { args: { entityType: "qualification" } };

export const Large: Story = { args: { size: 16 } };
