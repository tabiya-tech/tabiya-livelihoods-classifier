import type { Meta, StoryObj } from "@storybook/react";
import { ScoreBar } from "./ScoreBar";
import { ENTITY_TYPES } from "../../theme/theme";

const meta: Meta<typeof ScoreBar> = {
  title: "Primitives/ScoreBar",
  component: ScoreBar,
  parameters: { layout: "padded" },
  args: { score: 0.85 },
};
export default meta;

type Story = StoryObj<typeof ScoreBar>;

export const Default: Story = {};

export const FullSet: Story = {
  render: () => (
    <div className="flex flex-col gap-3 text-sm">
      {ENTITY_TYPES.map((type) => (
        <div key={type} className="flex items-center gap-3">
          <span className="w-28 font-mono text-xs text-muted">{type}</span>
          <ScoreBar score={0.6 + Math.random() * 0.35} entityType={type} />
        </div>
      ))}
    </div>
  ),
};

export const Wide: Story = { args: { score: 0.92, width: 160 } };

export const ScoreScale: Story = {
  render: () => (
    <div className="flex flex-col gap-3">
      {[0.2, 0.5, 0.75, 0.9, 1].map((value) => (
        <div key={value} className="flex items-center gap-3 font-mono text-xs">
          <span className="w-10 text-muted">{(value * 100).toFixed(0)}%</span>
          <ScoreBar score={value} />
        </div>
      ))}
    </div>
  ),
};
