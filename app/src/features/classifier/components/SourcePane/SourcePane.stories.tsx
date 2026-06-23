import { useState } from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import type { ClassifiedEntity, ClassifyEntityType } from "@/lib/api";
import {
  fixtureClassifyEntities,
  fixtureClassifySourceText,
} from "@/mocks/fixtures/classify";
import { ENTITY_TYPES, SourcePane } from "./SourcePane";

const meta: Meta<typeof SourcePane> = {
  title: "Features/Classifier/SourcePane",
  component: SourcePane,
  parameters: { layout: "padded" },
  args: {
    onTextChange: fn(),
    onRun: fn(),
    onClear: fn(),
    onSelectedEntityTypesChange: fn(),
    onTopKChange: fn(),
    onMinSimilarityChange: fn(),
    onEntitySelect: fn(),
    topK: 5,
    minSimilarity: 0,
    selectedEntityTypes: new Set(ENTITY_TYPES),
    isRunning: false,
  },
};
export default meta;

type Story = StoryObj<typeof SourcePane>;

export const EmptyTextarea: Story = {
  args: { text: "", entities: null, canRun: false },
};

export const WithText: Story = {
  args: { text: fixtureClassifySourceText, entities: null, canRun: true },
};

export const RunningRequest: Story = {
  args: {
    text: fixtureClassifySourceText,
    entities: null,
    canRun: true,
    isRunning: true,
  },
};

export const WithResults: Story = {
  args: {
    text: fixtureClassifySourceText,
    entities: fixtureClassifyEntities,
    canRun: true,
  },
};

/** Fully interactive — type, slide, toggle chips, run, clear. */
export const Interactive: Story = {
  render: function InteractiveStory(args) {
    const [text, setText] = useState(fixtureClassifySourceText);
    const [entities, setEntities] = useState<ClassifiedEntity[] | null>(null);
    const [topK, setTopK] = useState(args.topK);
    const [minSim, setMinSim] = useState(args.minSimilarity);
    const [selectedTypes, setSelectedTypes] = useState<
      Set<ClassifyEntityType>
    >(new Set(ENTITY_TYPES));
    const [running, setRunning] = useState(false);

    return (
      <SourcePane
        {...args}
        text={text}
        onTextChange={setText}
        entities={entities}
        topK={topK}
        minSimilarity={minSim}
        onTopKChange={setTopK}
        onMinSimilarityChange={setMinSim}
        selectedEntityTypes={selectedTypes}
        onSelectedEntityTypesChange={setSelectedTypes}
        canRun={text.trim().length > 0}
        isRunning={running}
        onRun={() => {
          setRunning(true);
          setTimeout(() => {
            setEntities(fixtureClassifyEntities);
            setRunning(false);
          }, 600);
        }}
        onClear={() => {
          setText("");
          setEntities(null);
        }}
      />
    );
  },
};
