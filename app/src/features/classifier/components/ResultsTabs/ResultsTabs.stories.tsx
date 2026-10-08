import { useState } from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { fixtureClassifyResponse } from "@/mocks/fixtures/classify";
import type { ClassifyEntityType } from "@/lib/api";
import { ENTITY_TYPES } from "../EntityTypeFilter/EntityTypeFilter";
import { ResultsTabs, type ResultsTabId } from "./ResultsTabs";

const meta: Meta<typeof ResultsTabs> = {
  title: "Features/Classifier/ResultsTabs",
  component: ResultsTabs,
  parameters: { layout: "padded" },
  args: {
    response: fixtureClassifyResponse,
    selectedEntityTypes: new Set<ClassifyEntityType>(ENTITY_TYPES),
    onActiveTabChange: fn(),
    onEntityClick: fn(),
  },
};
export default meta;

type Story = StoryObj<typeof ResultsTabs>;

export const EntitiesTab: Story = {
  args: { activeTabId: "entities" },
};

export const TableTab: Story = {
  args: { activeTabId: "table" },
};

export const JsonTab: Story = {
  args: { activeTabId: "json" },
};

export const OnlySkills: Story = {
  args: {
    activeTabId: "entities",
    selectedEntityTypes: new Set<ClassifyEntityType>(["skill"]),
  },
};

export const NothingSelected: Story = {
  args: {
    activeTabId: "entities",
    selectedEntityTypes: new Set(),
  },
};

/** Interactive — click any tab to see the panel switch. */
export const Interactive: Story = {
  render: function InteractiveStory(args) {
    const [tab, setTab] = useState<ResultsTabId>("entities");
    return (
      <ResultsTabs
        {...args}
        activeTabId={tab}
        onActiveTabChange={(next) => {
          setTab(next);
          args.onActiveTabChange(next);
        }}
      />
    );
  },
};
