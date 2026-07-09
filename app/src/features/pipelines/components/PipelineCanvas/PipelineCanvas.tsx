import { useCallback, useEffect, useRef, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  useReactFlow,
  addEdge,
} from "reactflow";
import type {
  Connection,
  NodeMouseHandler,
  OnConnect,
  ReactFlowInstance,
} from "reactflow";
import type {
  Pipeline,
  PipelineStage,
  PluginManifest,
  PipelineValidationIssue,
  PluginSummary,
} from "@/lib/api";
import { layoutStages } from "../../lib/layout";
import { areSlotsCompatible } from "../../lib/slotCompat";
import { StageNode } from "../StageNode/StageNode";

import "reactflow/dist/style.css";

const uniqueId = "d8e9f0a1-b2c3-d4e5-f6a7-b8c9d0e1f2a3";

export const DATA_TEST_ID = {
  ROOT: `pipeline-canvas-root-${uniqueId}`,
};

const NODE_TYPES = { stageNode: StageNode };

/** Data shape stored in drag-and-drop from the plugin palette. */
export interface PipelineDragData {
  pluginId: string;
  pluginSummary: PluginSummary;
}

export interface PipelineCanvasProps {
  pipeline: Pipeline;
  manifests: Record<string, PluginManifest>;
  validationIssues?: PipelineValidationIssue[];
  className?: string;
  editMode?: boolean;
  onStagesChange?: (nextStages: PipelineStage[]) => void;
  onConnectRejected?: (reason: string) => void;
  onNodeSelect?: (stageIndex: number) => void;
  /** Catalog summaries for the palette drop data (need category info). */
  pluginSummaries?: PluginSummary[];
}

function buildEnrichedNodes(
  pipeline: Pipeline,
  manifests: Record<string, PluginManifest>,
  validationIssues: PipelineValidationIssue[],
  shakingNodeIds: Set<string>,
) {
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
    className: shakingNodeIds.has(node.id) ? "animate-shake" : undefined,
  }));

  return { nodes, edges };
}

/**
 * ReadOnly variant — purely derived from props, no internal node/edge state.
 */
function ReadOnlyCanvas({
  pipeline,
  manifests,
  validationIssues,
  className,
}: Required<Pick<PipelineCanvasProps, "pipeline" | "manifests" | "validationIssues" | "className">>) {
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

/**
 * EditMode variant — manages controlled nodes/edges state and wires up
 * drag-and-drop, connection validation, node clicks.
 *
 * Requires a ReactFlowProvider ancestor (provided by the stories page/layout).
 */
function EditModeCanvas({
  pipeline,
  manifests,
  validationIssues,
  className,
  onStagesChange,
  onConnectRejected,
  onNodeSelect,
}: Required<Pick<PipelineCanvasProps, "pipeline" | "manifests" | "validationIssues" | "className" | "onStagesChange" | "onConnectRejected" | "onNodeSelect">>) {
  const [shakingNodeIds, setShakingNodeIds] = useState<Set<string>>(new Set());

  const { nodes: initialNodes, edges: initialEdges } = buildEnrichedNodes(
    pipeline,
    manifests,
    validationIssues,
    shakingNodeIds,
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const reactFlowInstance = useReactFlow<{ stageIndex: number; pluginId: string }>();

  // Re-initialise nodes/edges when the pipeline prop changes from the outside.
  const prevPipelineRef = useRef(pipeline);
  useEffect(() => {
    if (prevPipelineRef.current !== pipeline) {
      prevPipelineRef.current = pipeline;
      const { nodes: nextNodes, edges: nextEdges } = buildEnrichedNodes(
        pipeline,
        manifests,
        validationIssues,
        shakingNodeIds,
      );
      setNodes(nextNodes);
      setEdges(nextEdges);
    }
  }, [pipeline, manifests, validationIssues, shakingNodeIds, setNodes, setEdges]);

  // Apply shake className whenever shakingNodeIds changes.
  useEffect(() => {
    setNodes((currentNodes) =>
      currentNodes.map((node) => ({
        ...node,
        className: shakingNodeIds.has(node.id) ? "animate-shake" : undefined,
      })),
    );
  }, [shakingNodeIds, setNodes]);

  const triggerShake = useCallback(
    (nodeIds: string[]) => {
      setShakingNodeIds((prev) => {
        const next = new Set(prev);
        nodeIds.forEach((nodeId) => next.add(nodeId));
        return next;
      });
      setTimeout(() => {
        setShakingNodeIds((prev) => {
          const next = new Set(prev);
          nodeIds.forEach((nodeId) => next.delete(nodeId));
          return next;
        });
      }, 350);
    },
    [],
  );

  /**
   * Reconstruct the ordered PipelineStage list from current nodes in their
   * visual left-to-right order (sorted by x position).
   */
  const reconstructStages = useCallback(
    (currentNodes: typeof nodes): PipelineStage[] => {
      const sortedNodes = [...currentNodes].sort(
        (nodeA, nodeB) => nodeA.position.x - nodeB.position.x,
      );
      return sortedNodes.map((node) => {
        const stageIndex = node.data.stageIndex as number;
        return pipeline.stages[stageIndex] ?? { plugin_id: node.data.pluginId as string, config: {} };
      });
    },
    [pipeline.stages],
  );

  const handleNodesChange: typeof onNodesChange = useCallback(
    (changes) => {
      onNodesChange(changes);
      // After position changes settle, reconstruct stages from updated nodes.
      const hasPositionChange = changes.some(
        (change) => change.type === "position" && !change.dragging,
      );
      if (hasPositionChange) {
        setNodes((currentNodes) => {
          onStagesChange(reconstructStages(currentNodes));
          return currentNodes;
        });
      }
    },
    [onNodesChange, setNodes, onStagesChange, reconstructStages],
  );

  const handleEdgesChange: typeof onEdgesChange = useCallback(
    (changes) => {
      const hasRemoval = changes.some((change) => change.type === "remove");
      onEdgesChange(changes);
      if (hasRemoval) {
        setNodes((currentNodes) => {
          onStagesChange(reconstructStages(currentNodes));
          return currentNodes;
        });
      }
    },
    [onEdgesChange, setNodes, onStagesChange, reconstructStages],
  );

  const isValidConnection = useCallback(
    (connection: Connection): boolean => {
      const sourceNode = nodes.find((node) => node.id === connection.source);
      const targetNode = nodes.find((node) => node.id === connection.target);

      if (!sourceNode || !targetNode) {
        return false;
      }

      const sourceManifest = manifests[sourceNode.data.pluginId as string];
      const targetManifest = manifests[targetNode.data.pluginId as string];

      if (!sourceManifest || !targetManifest) {
        // Allow if manifests are unknown — cannot validate.
        return true;
      }

      const compatible = areSlotsCompatible(
        sourceManifest.output_slot.type,
        targetManifest.input_slot.type,
      );

      if (!compatible) {
        onConnectRejected("slot_mismatch");
        triggerShake([connection.source!, connection.target!]);
        return false;
      }

      return true;
    },
    [nodes, manifests, onConnectRejected, triggerShake],
  );

  const handleConnect: OnConnect = useCallback(
    (connection) => {
      if (!isValidConnection(connection)) {
        return;
      }
      setEdges((currentEdges) => addEdge(connection, currentEdges));
      // After connecting, reconstruct stage order from nodes sorted by x.
      setNodes((currentNodes) => {
        onStagesChange(reconstructStages(currentNodes));
        return currentNodes;
      });
    },
    [isValidConnection, setEdges, setNodes, onStagesChange, reconstructStages],
  );

  const handleNodeClick: NodeMouseHandler = useCallback(
    (_event, node) => {
      const stageIndex = node.data.stageIndex as number;
      onNodeSelect(stageIndex);
    },
    [onNodeSelect],
  );

  const handleDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  }, []);

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();

      const rawData = event.dataTransfer.getData("application/pipeline-plugin");
      if (!rawData) {
        return;
      }

      let dragData: PipelineDragData;
      try {
        dragData = JSON.parse(rawData) as PipelineDragData;
      } catch {
        return;
      }

      if (dragData.pluginSummary.coming_soon) {
        return;
      }

      const dropPosition = (reactFlowInstance as ReactFlowInstance).screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      // Determine insertion index based on drop X coordinate vs existing nodes.
      const sortedNodes = [...nodes].sort(
        (nodeA, nodeB) => nodeA.position.x - nodeB.position.x,
      );

      let insertionIndex = sortedNodes.length; // Default: append.
      for (let nodeIndex = 0; nodeIndex < sortedNodes.length - 1; nodeIndex++) {
        const leftNode = sortedNodes[nodeIndex];
        const rightNode = sortedNodes[nodeIndex + 1];
        const midpoint = (leftNode.position.x + rightNode.position.x) / 2;
        if (dropPosition.x < midpoint) {
          insertionIndex = nodeIndex + 1;
          break;
        }
      }
      if (sortedNodes.length > 0 && dropPosition.x < sortedNodes[0].position.x) {
        insertionIndex = 0;
      }

      const newStage: PipelineStage = {
        plugin_id: dragData.pluginId,
        config: {},
      };

      const currentStages = reconstructStages(sortedNodes.map((node) => ({
        ...node,
        data: { ...node.data },
      })));
      const nextStages = [
        ...currentStages.slice(0, insertionIndex),
        newStage,
        ...currentStages.slice(insertionIndex),
      ];

      onStagesChange(nextStages);
    },
    [nodes, reactFlowInstance, reconstructStages, onStagesChange],
  );

  return (
    <div
      data-testid={DATA_TEST_ID.ROOT}
      className={className}
      style={{ width: "100%", height: "400px" }}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={NODE_TYPES}
        nodesDraggable={true}
        nodesConnectable={true}
        elementsSelectable={true}
        onNodesChange={handleNodesChange}
        onEdgesChange={handleEdgesChange}
        isValidConnection={isValidConnection}
        onConnect={handleConnect}
        onNodeClick={handleNodeClick}
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

export function PipelineCanvas({
  pipeline,
  manifests,
  validationIssues = [],
  className = "",
  editMode = false,
  onStagesChange,
  onConnectRejected,
  onNodeSelect,
  pluginSummaries: _pluginSummaries,
}: PipelineCanvasProps) {
  if (editMode) {
    return (
      <EditModeCanvas
        pipeline={pipeline}
        manifests={manifests}
        validationIssues={validationIssues}
        className={className}
        onStagesChange={onStagesChange ?? (() => {})}
        onConnectRejected={onConnectRejected ?? (() => {})}
        onNodeSelect={onNodeSelect ?? (() => {})}
      />
    );
  }

  return (
    <ReadOnlyCanvas
      pipeline={pipeline}
      manifests={manifests}
      validationIssues={validationIssues}
      className={className}
    />
  );
}
