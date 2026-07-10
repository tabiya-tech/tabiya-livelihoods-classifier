import { useCallback, useEffect, useRef, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
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
import { layoutStages, buildConfigPreview } from "../../lib/layout";
import { areSlotsCompatible } from "../../lib/slotCompat";
import { defaultConfigForManifest } from "../../lib/defaultConfig";
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
  onDelete?: (stageIndex: number) => void,
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
      onDelete,
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
        fitViewOptions={{ padding: 0.3, maxZoom: 0.85 }}
        minZoom={0.3}
        maxZoom={1.25}
        proOptions={{ hideAttribution: true }}
      >
        <Background />
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

  // Remove a stage by index and hand the shortened list back to the parent.
  const handleStageDelete = useCallback(
    (stageIndex: number) => {
      const nextStages = pipeline.stages.filter(
        (_stage, index) => index !== stageIndex,
      );
      onStagesChange(nextStages);
    },
    [pipeline.stages, onStagesChange],
  );

  const { nodes: initialNodes, edges: initialEdges } = buildEnrichedNodes(
    pipeline,
    manifests,
    validationIssues,
    shakingNodeIds,
    handleStageDelete,
  );

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const reactFlowInstance = useReactFlow<{ stageIndex: number; pluginId: string }>();

  // Re-layout nodes/edges only when the pipeline STRUCTURE changes (stages
  // added, removed, or reordered) — keyed by the plugin_id sequence. A
  // config-only edit (same structure) must not re-run layout, or it would
  // reset any positions the user dragged. Validation-issue and manifest
  // changes are applied in a separate, position-preserving effect below.
  const structureKey = pipeline.stages.map((stage) => stage.plugin_id).join(">");
  const prevStructureKeyRef = useRef(structureKey);
  useEffect(() => {
    if (prevStructureKeyRef.current !== structureKey) {
      prevStructureKeyRef.current = structureKey;
      const { nodes: nextNodes, edges: nextEdges } = buildEnrichedNodes(
        pipeline,
        manifests,
        validationIssues,
        shakingNodeIds,
        handleStageDelete,
      );
      setNodes(nextNodes);
      setEdges(nextEdges);
    }
  }, [structureKey, pipeline, manifests, validationIssues, shakingNodeIds, handleStageDelete, setNodes, setEdges]);

  // Re-apply error/config data onto existing nodes WITHOUT moving them, so a
  // config edit or validation refresh updates node appearance in place.
  useEffect(() => {
    setNodes((currentNodes) =>
      currentNodes.map((node) => {
        const stageIndex = node.data.stageIndex as number;
        const stage = pipeline.stages[stageIndex];
        const hasError = validationIssues.some(
          (issue) => issue.stage_index === stageIndex,
        );
        return {
          ...node,
          data: {
            ...node.data,
            hasError,
            configPreview: stage ? buildConfigPreview(stage) : node.data.configPreview,
          },
        };
      }),
    );
    // structureKey guards against running during a full re-layout tick.
  }, [validationIssues, pipeline.stages, structureKey, setNodes]);

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
      // After a drag settles, only propagate to the parent when the drag
      // actually changed the left-right STAGE ORDER. A plain nudge that keeps
      // the order must NOT trigger onStagesChange — doing so rebuilds the
      // pipeline prop, which re-runs layout and snaps the node back to the
      // grid (the "jumps to center" bug). Positions aren't part of the
      // pipeline model, so a nudge that preserves order is a no-op upstream.
      const hasPositionChange = changes.some(
        (change) => change.type === "position" && !change.dragging,
      );
      if (hasPositionChange) {
        setNodes((currentNodes) => {
          const nextStages = reconstructStages(currentNodes);
          const orderChanged =
            nextStages.length !== pipeline.stages.length ||
            nextStages.some(
              (stage, index) =>
                stage.plugin_id !== pipeline.stages[index]?.plugin_id,
            );
          if (orderChanged) {
            onStagesChange(nextStages);
          }
          return currentNodes;
        });
      }
    },
    [onNodesChange, setNodes, onStagesChange, reconstructStages, pipeline.stages],
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

      // Seed config from the manifest's declared defaults so optional fields
      // (top_k, min_similarity, text_field, …) are populated immediately.
      // Required fields with no default stay absent and are flagged by the
      // required-field UI until the user fills them.
      const newStage: PipelineStage = {
        plugin_id: dragData.pluginId,
        config: defaultConfigForManifest(manifests[dragData.pluginId]),
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
    [nodes, manifests, reactFlowInstance, reconstructStages, onStagesChange],
  );

  return (
    <div
      data-testid={DATA_TEST_ID.ROOT}
      className={className}
      // Fill the parent (the editor's flex-1 canvas area) so React Flow's
      // bottom-right Controls sit at the true bottom-right of the canvas,
      // not mid-page (which a fixed height would cause).
      style={{ width: "100%", height: "100%", minHeight: "400px" }}
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
        fitViewOptions={{ padding: 0.3, maxZoom: 0.85 }}
        minZoom={0.3}
        maxZoom={1.25}
        proOptions={{ hideAttribution: true }}
      >
        <Background />
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
