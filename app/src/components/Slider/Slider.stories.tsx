import type { Meta, StoryObj } from "@storybook/react";
import type { ChangeEvent } from "react";
import { useState } from "react";
import { fn } from "@storybook/test";
import { Slider } from "./Slider";

const meta: Meta<typeof Slider> = {
  title: "Primitives/Slider",
  component: Slider,
  parameters: { layout: "padded" },
};
export default meta;

type Story = StoryObj<typeof Slider>;

export const TopK: Story = {
  render: () => {
    const [topK, setTopK] = useState(3);
    const onChange = fn((event: ChangeEvent<HTMLInputElement>) =>
      setTopK(Number(event.target.value)),
    );
    return (
      <Slider
        label="top_k"
        hint="Max ESCO matches per entity"
        min={1}
        max={10}
        value={topK}
        onChange={onChange}
      />
    );
  },
};

export const MinSimilarity: Story = {
  render: () => {
    const [minSimilarity, setMinSimilarity] = useState(0.5);
    const onChange = fn((event: ChangeEvent<HTMLInputElement>) =>
      setMinSimilarity(Number(event.target.value)),
    );
    return (
      <Slider
        label="min_similarity"
        hint="Filter weak matches"
        min={0}
        max={1}
        step={0.05}
        value={minSimilarity}
        format={(value) => value.toFixed(2)}
        onChange={onChange}
      />
    );
  },
};
