import { useState } from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import {
  fixtureClassifyEntities,
  fixtureClassifySourceText,
} from "@/mocks/fixtures/classify";
import { EntityHighlight } from "./EntityHighlight";

const meta: Meta<typeof EntityHighlight> = {
  title: "Features/Classifier/EntityHighlight",
  component: EntityHighlight,
  parameters: { layout: "padded" },
  args: {
    text: fixtureClassifySourceText,
    entities: fixtureClassifyEntities,
    onEntityClick: fn(),
  },
};
export default meta;

type Story = StoryObj<typeof EntityHighlight>;

/** Static — five entities of three different types. */
export const Default: Story = {};

/** Second entity rendered as selected (focus ring). */
export const WithSelection: Story = {
  args: { selectedEntityIndex: 1 },
};

/** Skills dimmed, occupation and qualification rendered normally. */
export const SkillsDimmed: Story = {
  args: {
    dimmedEntityIndices: new Set(
      fixtureClassifyEntities
        .map((entity, index) => (entity.entity_type === "skill" ? index : -1))
        .filter((index) => index >= 0),
    ),
  },
};

/**
 * Interactive harness — click any entity to toggle its selection. Useful
 * for visually checking the selected-ring style.
 */
export const Interactive: Story = {
  render: function InteractiveStory(args) {
    const [selected, setSelected] = useState<number | null>(null);
    return (
      <EntityHighlight
        {...args}
        selectedEntityIndex={selected}
        onEntityClick={(_entity, entityIndex) => {
          setSelected((current) => (current === entityIndex ? null : entityIndex));
          args.onEntityClick?.(_entity, entityIndex);
        }}
      />
    );
  },
};

/**
 * Overlap-policy demo — two entities cover the same span, the higher-scoring
 * one is kept and the lower is dropped silently.
 */
export const OverlapResolved: Story = {
  args: {
    text: "Senior data scientist role",
    entities: [
      {
        entity_type: "occupation",
        surface_form: "data scientist",
        span: { start: 7, end: 21 },
        matches: [
          {
            entity_type: "occupation",
            similarity_score: 0.95,
            entity: {
              uuid: "kept",
              origin_uuid: "kept",
              uuid_history: [],
              preferred_label: "data scientist (kept)",
              origin_uri: "",
              alt_labels: [],
              description: "",
            },
          },
        ],
      },
      {
        entity_type: "skill",
        surface_form: "scientist",
        span: { start: 12, end: 21 },
        matches: [
          {
            entity_type: "skill",
            similarity_score: 0.6,
            entity: {
              uuid: "dropped",
              origin_uuid: "dropped",
              uuid_history: [],
              preferred_label: "scientist (dropped)",
              origin_uri: "",
              alt_labels: [],
              description: "",
            },
          },
        ],
      },
    ],
  },
};
