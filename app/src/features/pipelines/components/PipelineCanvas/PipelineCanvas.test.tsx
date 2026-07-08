import { describe, expect, it } from "vitest";
import { layoutStages } from "../../lib/layout";
import type { PipelineStage } from "@/lib/api";
import {
  fixturePluginManifests,
  fixtureTextInputManifest,
  fixtureNerManifest,
  fixtureNelManifest,
  fixtureResultsManifest,
} from "@/mocks/fixtures/plugins";

const givenDefaultStages: PipelineStage[] = [
  { plugin_id: "tabiya.source.text.v1", config: { text: "" } },
  { plugin_id: "tabiya.ner.v1", config: {} },
  { plugin_id: "tabiya.nel.v1", config: { nel_model_id: "all-MiniLM-L6-v2", taxonomy_model_id: "esco-v1.2", top_k: 5, min_similarity: 0.0 } },
  { plugin_id: "tabiya.sink.results.v1", config: {} },
];

describe("PipelineCanvas layout (via layoutStages helper)", () => {
  it("produces one node per stage and one edge between each adjacent pair", () => {
    // GIVEN a 4-stage pipeline with manifests
    const expectedNodeCount = 4;
    const expectedEdgeCount = 3;

    // WHEN we compute the layout
    const { nodes, edges } = layoutStages(givenDefaultStages, fixturePluginManifests);

    // THEN node and edge counts match the stage count
    expect(nodes).toHaveLength(expectedNodeCount);
    expect(edges).toHaveLength(expectedEdgeCount);
  });

  it("assigns compatible slot types to adjacent stages in the default pipeline", () => {
    // GIVEN the default pipeline manifests with compatible slot types
    const expectedCompatibleEdgeCount = 3;

    // WHEN we compute the layout
    const { edges } = layoutStages(givenDefaultStages, fixturePluginManifests);

    // THEN none of the edges have the error stroke style (all slots match)
    const errorEdges = edges.filter(
      (edge) => edge.style?.stroke === "#c0392b",
    );
    expect(errorEdges).toHaveLength(0);
    expect(edges).toHaveLength(expectedCompatibleEdgeCount);
  });

  it("marks an edge as error when slot types are incompatible", () => {
    // GIVEN a mismatched pipeline: skip NER so text_input->nel has RawText->Entities mismatch
    const givenMismatchedStages: PipelineStage[] = [
      { plugin_id: fixtureTextInputManifest.plugin_id, config: {} },
      { plugin_id: fixtureNelManifest.plugin_id, config: {} },
      { plugin_id: fixtureResultsManifest.plugin_id, config: {} },
    ];

    // WHEN we compute the layout with full manifests
    const { edges } = layoutStages(givenMismatchedStages, {
      [fixtureTextInputManifest.plugin_id]: fixtureTextInputManifest,
      [fixtureNerManifest.plugin_id]: fixtureNerManifest,
      [fixtureNelManifest.plugin_id]: fixtureNelManifest,
      [fixtureResultsManifest.plugin_id]: fixtureResultsManifest,
    });

    // THEN the first edge (text_input->nel) is colored error
    const errorEdge = edges.find((edge) => edge.style?.stroke === "#c0392b");
    expect(errorEdge).toBeDefined();
    expect(errorEdge?.id).toBe("edge-0-1");
  });

  it("places each node at stageIndex * 240 on the x axis", () => {
    // GIVEN 4 stages
    const expectedXPositions = [0, 240, 480, 720];

    // WHEN we compute the layout
    const { nodes } = layoutStages(givenDefaultStages, fixturePluginManifests);

    // THEN every node is at the correct x position
    nodes.forEach((node, stageIndex) => {
      expect(node.position.x).toBe(expectedXPositions[stageIndex]);
    });
  });
});
