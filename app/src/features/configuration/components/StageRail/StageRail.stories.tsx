import type { Meta, StoryObj } from "@storybook/react";
import { useState } from "react";
import { fn } from "@storybook/test";
import { StageRail } from "./StageRail";

const meta: Meta<typeof StageRail> = {
  title: "Features/Configuration/StageRail",
  component: StageRail,
  parameters: { layout: "padded" },
};
export default meta;

type Story = StoryObj<typeof StageRail>;

const stages = [
  {
    id: "nel",
    number: "01",
    label: "NEL",
    subLabel: "Entity linking",
    currentValue: "MPNet base v2",
  },
  {
    id: "taxonomy",
    number: "02",
    label: "Taxonomy",
    subLabel: "Reference vocabulary",
    currentValue: "ESCO v1.2.0",
  },
];

export const NelActive: Story = {
  args: {
    items: stages,
    activeId: "nel",
    onSelect: fn(),
  },
};

export const TaxonomyActive: Story = {
  args: {
    items: stages,
    activeId: "taxonomy",
    onSelect: fn(),
  },
};

/**
 * The rail wired to local state so clicking visibly swaps the active card.
 * Mirrors how the Configuration page composes it.
 */
export const Interactive: Story = {
  render: () => {
    const [activeId, setActiveId] = useState<string>("nel");
    const onSelect = fn((nextId: string) => setActiveId(nextId));
    return (
      <div className="w-72">
        <StageRail
          items={stages}
          activeId={activeId}
          onSelect={onSelect}
          aria-label="Pipeline stages"
        />
      </div>
    );
  },
};
