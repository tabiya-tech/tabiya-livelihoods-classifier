import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReactFlowProvider } from "reactflow";
import type { PluginManifest } from "@/lib/api";
import { StageNode, DATA_TEST_ID } from "./StageNode";

const givenManifest: PluginManifest = {
  plugin_id: "tabiya.ner.v1",
  name: "Tabiya NER",
  version: "0.1.0",
  category: "core",
  summary: "Named-entity recognition over job-ad prose.",
  icon: "ner",
  input_slot: { type: "RawText", cardinality: "single" },
  output_slot: { type: "Entities", cardinality: "single" },
  config_schema: {},
  timeout_ms: 30_000,
};

const givenBaseNodeProps = {
  id: "stage-1",
  type: "stageNode",
  selected: false,
  isConnectable: false,
  zIndex: 1,
  xPos: 240,
  yPos: 0,
  dragging: false,
};

describe("StageNode", () => {
  it("renders the plugin name from the manifest", () => {
    // GIVEN a node with a manifest
    const givenData = {
      pluginId: "tabiya.ner.v1",
      manifest: givenManifest,
      stageIndex: 1,
    };
    const expectedName = "Tabiya NER";

    // WHEN we render
    render(
      <ReactFlowProvider>
        <StageNode {...givenBaseNodeProps} data={givenData} />
      </ReactFlowProvider>,
    );

    // THEN the plugin name is visible
    expect(screen.getByTestId(DATA_TEST_ID.PLUGIN_NAME)).toHaveTextContent(
      expectedName,
    );
  });

  it("falls back to plugin_id when manifest is absent", () => {
    // GIVEN a node with no manifest
    const givenPluginId = "tabiya.unknown.v1";
    const givenData = {
      pluginId: givenPluginId,
      stageIndex: 0,
    };

    // WHEN we render
    render(
      <ReactFlowProvider>
        <StageNode {...givenBaseNodeProps} data={givenData} />
      </ReactFlowProvider>,
    );

    // THEN the plugin_id is used as fallback
    expect(screen.getByTestId(DATA_TEST_ID.PLUGIN_NAME)).toHaveTextContent(
      givenPluginId,
    );
  });

  it("renders slot pills showing input and output slot types", () => {
    // GIVEN a node with a manifest with known slot types
    const givenData = {
      pluginId: "tabiya.ner.v1",
      manifest: givenManifest,
      stageIndex: 1,
    };
    const expectedInputSlot = "RawText";
    const expectedOutputSlot = "Entities";

    // WHEN we render
    render(
      <ReactFlowProvider>
        <StageNode {...givenBaseNodeProps} data={givenData} />
      </ReactFlowProvider>,
    );

    // THEN both slot pills show the correct type labels
    expect(screen.getByTestId(DATA_TEST_ID.INPUT_SLOT_PILL)).toHaveTextContent(
      expectedInputSlot,
    );
    expect(screen.getByTestId(DATA_TEST_ID.OUTPUT_SLOT_PILL)).toHaveTextContent(
      expectedOutputSlot,
    );
  });

  it("renders the configPreview text when provided", () => {
    // GIVEN a node with a configPreview
    const givenConfigPreview = "top_k=5, min_similarity=0.4";
    const givenData = {
      pluginId: "tabiya.nel.v1",
      manifest: givenManifest,
      stageIndex: 2,
      configPreview: givenConfigPreview,
    };

    // WHEN we render
    render(
      <ReactFlowProvider>
        <StageNode {...givenBaseNodeProps} data={givenData} />
      </ReactFlowProvider>,
    );

    // THEN the config preview is shown
    expect(screen.getByTestId(DATA_TEST_ID.CONFIG_PREVIEW)).toHaveTextContent(
      givenConfigPreview,
    );
  });

  it("renders the validation badge when hasError is true", () => {
    // GIVEN a node with hasError set
    const givenData = {
      pluginId: "tabiya.ner.v1",
      stageIndex: 1,
      hasError: true,
    };

    // WHEN we render
    render(
      <ReactFlowProvider>
        <StageNode {...givenBaseNodeProps} data={givenData} />
      </ReactFlowProvider>,
    );

    // THEN the validation badge container is present
    expect(screen.getByTestId(DATA_TEST_ID.VALIDATION_BADGE)).toBeInTheDocument();
  });

  it("does not render the validation badge when hasError is false", () => {
    // GIVEN a node without hasError
    const givenData = {
      pluginId: "tabiya.ner.v1",
      stageIndex: 1,
      hasError: false,
    };

    // WHEN we render
    render(
      <ReactFlowProvider>
        <StageNode {...givenBaseNodeProps} data={givenData} />
      </ReactFlowProvider>,
    );

    // THEN no validation badge is present
    expect(screen.queryByTestId(DATA_TEST_ID.VALIDATION_BADGE)).toBeNull();
  });

  it("renders a delete button that calls onDelete with the stage index", async () => {
    // GIVEN a node with an onDelete handler
    const deleteCalls: number[] = [];
    const givenStageIndex = 2;
    const givenData = {
      pluginId: "tabiya.nel.v1",
      manifest: givenManifest,
      stageIndex: givenStageIndex,
      onDelete: (index: number) => deleteCalls.push(index),
    };

    // WHEN we render and click the delete button
    render(
      <ReactFlowProvider>
        <StageNode {...givenBaseNodeProps} data={givenData} />
      </ReactFlowProvider>,
    );
    screen.getByTestId(DATA_TEST_ID.DELETE_BUTTON).click();

    // THEN onDelete fires with this node's stage index
    expect(deleteCalls).toEqual([givenStageIndex]);
  });

  it("does not render a delete button when onDelete is absent (read-only)", () => {
    // GIVEN a node with no onDelete handler
    const givenData = {
      pluginId: "tabiya.ner.v1",
      manifest: givenManifest,
      stageIndex: 0,
    };

    // WHEN we render
    render(
      <ReactFlowProvider>
        <StageNode {...givenBaseNodeProps} data={givenData} />
      </ReactFlowProvider>,
    );

    // THEN there is no delete affordance
    expect(screen.queryByTestId(DATA_TEST_ID.DELETE_BUTTON)).toBeNull();
  });

  it("uses a non-gear icon derived from the manifest (not the config/sun default)", () => {
    // GIVEN a NER manifest whose icon should map to 'search', not 'config'
    const givenData = {
      pluginId: "tabiya.ner.v1",
      manifest: givenManifest,
      stageIndex: 1,
    };

    // WHEN we render
    const { container } = render(
      <ReactFlowProvider>
        <StageNode {...givenBaseNodeProps} data={givenData} />
      </ReactFlowProvider>,
    );

    // THEN the header icon is not the gear/sun ('config') default
    const configIcon = container.querySelector('[data-icon="config"]');
    expect(configIcon).toBeNull();
  });
});
