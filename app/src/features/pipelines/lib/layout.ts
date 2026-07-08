import type { Node, Edge } from "reactflow";
import type { PipelineStage, PluginManifest } from "@/lib/api";
import { areSlotsCompatible } from "./slotCompat";

const HORIZONTAL_SPACING_PX = 240;
const NODE_WIDTH_PX = 200;
const NODE_HEIGHT_PX = 80;
const VERTICAL_CENTER_Y = 0;

export interface StageNodeData {
  pluginId: string;
  manifest?: PluginManifest;
  stageIndex: number;
  configPreview?: string;
  hasError?: boolean;
}

export type LayoutedNode = Node<StageNodeData>;
export type LayoutedEdge = Edge;

export interface LayoutResult {
  nodes: LayoutedNode[];
  edges: LayoutedEdge[];
}

/**
 * Converts a flat stage list into React Flow nodes and edges positioned
 * left-to-right. Each node's x = stageIndex * HORIZONTAL_SPACING_PX; y = 0.
 */
export function layoutStages(
  stages: PipelineStage[],
  manifests: Record<string, PluginManifest> = {},
): LayoutResult {
  const nodes: LayoutedNode[] = stages.map((stage, stageIndex) => ({
    id: `stage-${stageIndex}`,
    type: "stageNode",
    position: { x: stageIndex * HORIZONTAL_SPACING_PX, y: VERTICAL_CENTER_Y },
    data: {
      pluginId: stage.plugin_id,
      manifest: manifests[stage.plugin_id],
      stageIndex,
      configPreview: buildConfigPreview(stage),
    },
    width: NODE_WIDTH_PX,
    height: NODE_HEIGHT_PX,
  }));

  const edges: LayoutedEdge[] = stages.slice(0, -1).map((stage, stageIndex) => {
    const nextStage = stages[stageIndex + 1];
    const sourceManifest = manifests[stage.plugin_id];
    const targetManifest = manifests[nextStage.plugin_id];

    const slotsMatch =
      sourceManifest && targetManifest
        ? areSlotsCompatible(
            sourceManifest.output_slot.type,
            targetManifest.input_slot.type,
          )
        : true;

    return {
      id: `edge-${stageIndex}-${stageIndex + 1}`,
      source: `stage-${stageIndex}`,
      target: `stage-${stageIndex + 1}`,
      style: slotsMatch ? undefined : { stroke: "#c0392b", strokeWidth: 2 },
    };
  });

  return { nodes, edges };
}

function buildConfigPreview(stage: PipelineStage): string {
  const entries = Object.entries(stage.config)
    .filter(([, value]) => value !== "" && value !== null && value !== undefined)
    .slice(0, 2)
    .map(([key, value]) => `${key}=${String(value)}`);
  return entries.join(", ");
}
