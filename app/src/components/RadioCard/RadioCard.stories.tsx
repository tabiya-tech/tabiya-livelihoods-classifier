import type { Meta, StoryObj } from "@storybook/react";
import { useState } from "react";
import { fn } from "@storybook/test";
import { RadioCard } from "./RadioCard";
import { Tag } from "@/components";

const meta: Meta<typeof RadioCard> = {
  title: "Primitives/RadioCard",
  component: RadioCard,
  parameters: { layout: "padded" },
};
export default meta;

type Story = StoryObj<typeof RadioCard>;

const models = [
  { id: "roberta", name: "RoBERTa base (job-NER)", desc: "Default. Trained on multilingual job postings.", params: "125M", recommended: true },
  { id: "deberta", name: "DeBERTa-v3 (job-NER)", desc: "Highest precision; ~1.6× slower.", params: "184M" },
  { id: "distil", name: "DistilBERT (job-NER)", desc: "Fastest. Use for high-throughput batches.", params: "66M" },
];

export const SingleSelect: Story = {
  render: () => {
    const [pickedModelId, setPickedModelId] = useState("roberta");
    const onPick = fn((modelId: string) => setPickedModelId(modelId));
    return (
      <div role="radiogroup" className="flex max-w-xl flex-col gap-2">
        {models.map((model) => (
          <RadioCard
            key={model.id}
            selected={pickedModelId === model.id}
            onClick={() => onPick(model.id)}
            title={model.name}
            description={model.desc}
            meta={
              <>
                <Tag size="sm">{model.params} params</Tag>
                {model.recommended && (
                  <Tag size="sm" tone="lime">
                    recommended
                  </Tag>
                )}
              </>
            }
          />
        ))}
      </div>
    );
  },
};
