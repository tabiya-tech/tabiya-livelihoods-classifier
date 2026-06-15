import type { Meta, StoryObj } from "@storybook/react";
import { useState } from "react";
import { fn } from "@storybook/test";
import { ModelOption } from "./ModelOption";

const meta: Meta<typeof ModelOption> = {
  title: "Features/Configuration/ModelOption",
  component: ModelOption,
  parameters: { layout: "padded" },
  args: { onClick: fn() },
};
export default meta;

type Story = StoryObj<typeof ModelOption>;

export const Unselected: Story = {
  args: {
    title: "MPNet base v2",
    description:
      "Higher quality embeddings; ~2× slower. Recommended for analyst workflows.",
    suffix: "768-dim",
  },
};

export const Selected: Story = {
  args: {
    title: "MPNet base v2",
    description:
      "Higher quality embeddings; ~2× slower. Recommended for analyst workflows.",
    suffix: "768-dim",
    selected: true,
  },
};

export const Recommended: Story = {
  args: {
    title: "RoBERTa base (job-NER)",
    description:
      "Default. Trained on multilingual job postings; reliable across occupations and skills.",
    suffix: "125M params",
    recommended: true,
  },
};

export const NoMeta: Story = {
  args: {
    title: "Plain option",
    description: "No suffix tag, no recommended badge.",
  },
};

/**
 * A small group rendered with shared state, matching how the Configuration
 * page composes a list of options into a radio group.
 */
export const RadioGroup: Story = {
  render: () => {
    const models = [
      {
        modelId: "all-MiniLM-L6-v2",
        title: "MiniLM v3",
        description: "Fast general-purpose sentence embedder.",
        suffix: "384-dim",
      },
      {
        modelId: "mpnet-base-v2",
        title: "MPNet base v2",
        description: "Higher quality embeddings; ~2× slower.",
        suffix: "768-dim",
        recommended: true,
      },
      {
        modelId: "tabiya-job-bge",
        title: "Tabiya Job-BGE",
        description: "Fine-tuned on job-ad corpora.",
        suffix: "1024-dim",
      },
    ];
    const [selectedModelId, setSelectedModelId] =
      useState<string>("mpnet-base-v2");
    const onPick = fn((modelId: string) => setSelectedModelId(modelId));
    return (
      <div role="radiogroup" className="flex max-w-xl flex-col gap-2">
        {models.map((model) => (
          <ModelOption
            key={model.modelId}
            modelId={model.modelId}
            title={model.title}
            description={model.description}
            suffix={model.suffix}
            recommended={model.recommended}
            selected={selectedModelId === model.modelId}
            onClick={() => onPick(model.modelId)}
          />
        ))}
      </div>
    );
  },
};
