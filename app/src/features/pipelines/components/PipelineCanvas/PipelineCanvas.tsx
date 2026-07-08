import ReactFlow, {
  Background,
  Controls,
  MiniMap,
} from "reactflow";
import type { Pipeline, PluginManifest, PipelineValidationIssue } from "@/lib/api";
import { layoutStages } from "../../lib/layout";
import { StageNode } from "../StageNode/StageNode";

import "reactflow/dist/style.css";

const uniqueId = "d8e9f0a1-b2c3-d4e5-f6a7-b8c9d0e1f2a3";

export const DATA_TEST_ID = {
  ROOT: `pipeline-canvas-root-${uniqueId}`,
};

const NODE_TYPES = { stageNode: StageNode };

export interface PipelineCanvasProps {
  pipeline: Pipeline;
  manifests: Record<string, PluginManifest>;
  validationIssues?: PipelineValidationIssue[];
  className?: string;
}

export function PipelineCanvas({
  pipeline,
  manifests,
  validationIssues = [],
  className,
}: PipelineCanvasProps) {
  const stagesWithErrors = new Set(
    validationIssues
      .filter((issue) => issue.stage_index != null)
      .map((issue) => issue.stage_index as number),
  );

  const { nodes: layoutedNodes, edges } = layoutStages(pipeline.stages, manifests);

  const nodes = layoutedNodes.map((node) => ({
    ...node,
    data: {
      ...node.data,
      hasError: stagesWithErrors.has(node.data.stageIndex),
    },
  }));

  return (
    <div
      data-testid={DATA_TEST_ID.ROOT}
      className={className}
      style={{ width: "100%", height: "400px" }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={NODE_TYPES}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={true}
        fitView
        fitViewOptions={{ padding: 0.2 }}
      >
        <Background />
        <MiniMap />
        <Controls position="bottom-right" />
      </ReactFlow>
    </div>
  );
}
