import { describe, expect, it } from "vitest";
import { layoutStages } from "./layout";
import type { PipelineStage } from "@/lib/api";

const givenTwoStages: PipelineStage[] = [
  { plugin_id: "tabiya.source.text.v1", config: {} },
  { plugin_id: "tabiya.sink.results.v1", config: {} },
];

const givenFourStages: PipelineStage[] = [
  { plugin_id: "tabiya.source.text.v1", config: {} },
  { plugin_id: "tabiya.ner.v1", config: {} },
  { plugin_id: "tabiya.nel.v1", config: {} },
  { plugin_id: "tabiya.sink.results.v1", config: {} },
];

const givenSixStages: PipelineStage[] = [
  { plugin_id: "tabiya.source.text.v1", config: {} },
  { plugin_id: "tabiya.ner.v1", config: {} },
  { plugin_id: "tabiya.nel.v1", config: {} },
  { plugin_id: "tabiya.sink.results.v1", config: {} },
  { plugin_id: "extra.stage.one.v1", config: {} },
  { plugin_id: "extra.stage.two.v1", config: {} },
];

describe("layoutStages", () => {
  it("lays out 2 stages left-to-right with correct x positions", () => {
    // GIVEN two pipeline stages
    const expectedNodeCount = 2;
    const expectedEdgeCount = 1;

    // WHEN we compute the layout
    const { nodes, edges } = layoutStages(givenTwoStages);

    // THEN we get the right node count with left-to-right positions
    expect(nodes).toHaveLength(expectedNodeCount);
    expect(edges).toHaveLength(expectedEdgeCount);
    expect(nodes[0].position.x).toBe(0);
    expect(nodes[1].position.x).toBe(240);
    expect(nodes[0].position.y).toBe(0);
    expect(nodes[1].position.y).toBe(0);
  });

  it("lays out 4 stages with each node's x = stageIndex * 240", () => {
    // GIVEN four pipeline stages
    const expectedNodeCount = 4;
    const expectedEdgeCount = 3;
    const expectedXPositions = [0, 240, 480, 720];

    // WHEN we compute the layout
    const { nodes, edges } = layoutStages(givenFourStages);

    // THEN each node is positioned at stageIndex * 240, y = 0
    expect(nodes).toHaveLength(expectedNodeCount);
    expect(edges).toHaveLength(expectedEdgeCount);
    nodes.forEach((node, stageIndex) => {
      expect(node.position.x).toBe(expectedXPositions[stageIndex]);
      expect(node.position.y).toBe(0);
    });
  });

  it("lays out 6 stages with all y = 0", () => {
    // GIVEN six pipeline stages
    const expectedNodeCount = 6;
    const expectedEdgeCount = 5;

    // WHEN we compute the layout
    const { nodes, edges } = layoutStages(givenSixStages);

    // THEN all nodes have y = 0 and x = stageIndex * 240
    expect(nodes).toHaveLength(expectedNodeCount);
    expect(edges).toHaveLength(expectedEdgeCount);
    nodes.forEach((node, stageIndex) => {
      expect(node.position.x).toBe(stageIndex * 240);
      expect(node.position.y).toBe(0);
    });
  });

  it("assigns stageNode type and correct data to each node", () => {
    // GIVEN two stages
    const expectedFirstPluginId = "tabiya.source.text.v1";

    // WHEN we compute the layout
    const { nodes } = layoutStages(givenTwoStages);

    // THEN nodes carry the correct data and type
    expect(nodes[0].type).toBe("stageNode");
    expect(nodes[0].data.pluginId).toBe(expectedFirstPluginId);
    expect(nodes[0].data.stageIndex).toBe(0);
  });
});
