import { useState } from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { PipelineParamsCard } from "./PipelineParamsCard";

const meta: Meta<typeof PipelineParamsCard> = {
  title: "Features/Classifier/PipelineParamsCard",
  component: PipelineParamsCard,
  parameters: { layout: "padded" },
  args: {
    topK: 5,
    minSimilarity: 0,
    onTopKChange: fn(),
    onMinSimilarityChange: fn(),
  },
};
export default meta;

type Story = StoryObj<typeof PipelineParamsCard>;

export const Default: Story = {};

export const Disabled: Story = {
  args: { disabled: true },
};

/** Interactive harness so you can play with the sliders end-to-end. */
export const Interactive: Story = {
  render: function InteractiveStory() {
    const [topK, setTopK] = useState(3);
    const [minSim, setMinSim] = useState(0.4);
    return (
      <PipelineParamsCard
        topK={topK}
        minSimilarity={minSim}
        onTopKChange={setTopK}
        onMinSimilarityChange={setMinSim}
      />
    );
  },
};
