import { useState } from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import type { ClassifyEntityType } from "@/lib/api";
import { ENTITY_TYPES, EntityTypeFilter } from "./EntityTypeFilter";

const meta: Meta<typeof EntityTypeFilter> = {
  title: "Features/Classifier/EntityTypeFilter",
  component: EntityTypeFilter,
  parameters: { layout: "padded" },
  args: {
    counts: { occupation: 1, skill: 3, qualification: 1 },
    selected: new Set(ENTITY_TYPES),
    onChange: fn(),
  },
};
export default meta;

type Story = StoryObj<typeof EntityTypeFilter>;

export const AllSelected: Story = {};

export const OnlySkills: Story = {
  args: { selected: new Set<ClassifyEntityType>(["skill"]) },
};

export const Disabled: Story = {
  args: { disabled: true },
};

/** Toggle chips and see selected state flip. */
export const Interactive: Story = {
  render: function InteractiveStory(args) {
    const [selected, setSelected] = useState<Set<ClassifyEntityType>>(
      new Set(ENTITY_TYPES),
    );
    return (
      <EntityTypeFilter
        {...args}
        selected={selected}
        onChange={(next) => {
          setSelected(next);
          args.onChange(next);
        }}
      />
    );
  },
};
