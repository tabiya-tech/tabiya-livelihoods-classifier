import { Handle, Position } from "reactflow";
import type { NodeProps } from "reactflow";
import type { PluginManifest, PluginStatus } from "@/lib/api";
import { Icon } from "@/components/Icon/Icon";
import { ValidationBadge } from "../ValidationBadge/ValidationBadge";
import { slotColor } from "../../lib/slotCompat";

const uniqueId = "c7d1e2f3-4a5b-6c7d-8e9f-0a1b2c3d4e5f";

export const DATA_TEST_ID = {
  ROOT: `stage-node-root-${uniqueId}`,
  PLUGIN_NAME: `stage-node-plugin-name-${uniqueId}`,
  CONFIG_PREVIEW: `stage-node-config-preview-${uniqueId}`,
  INPUT_HANDLE: `stage-node-input-handle-${uniqueId}`,
  OUTPUT_HANDLE: `stage-node-output-handle-${uniqueId}`,
  INPUT_SLOT_PILL: `stage-node-input-slot-pill-${uniqueId}`,
  OUTPUT_SLOT_PILL: `stage-node-output-slot-pill-${uniqueId}`,
  STATUS_INDICATOR: `stage-node-status-indicator-${uniqueId}`,
  VALIDATION_BADGE: `stage-node-validation-badge-${uniqueId}`,
};

export interface StageNodeData {
  pluginId: string;
  manifest?: PluginManifest;
  status?: PluginStatus;
  stageIndex: number;
  configPreview?: string;
  hasError?: boolean;
}

const STATUS_COLORS: Record<PluginStatus, string> = {
  enabled: "#00d579",
  degraded: "#eeff41",
  unavailable: "#c0392b",
};

export function StageNode({ data }: NodeProps<StageNodeData>) {
  const { manifest, pluginId, status, stageIndex, configPreview, hasError } = data;

  const pluginName = manifest?.name ?? pluginId;
  const inputSlotType = manifest?.input_slot.type ?? "None";
  const outputSlotType = manifest?.output_slot.type ?? "None";
  const inputColor = slotColor(inputSlotType);
  const outputColor = slotColor(outputSlotType);
  const statusColor = status ? STATUS_COLORS[status] : "#c9c5be";

  return (
    <div
      data-testid={DATA_TEST_ID.ROOT}
      data-stage-index={stageIndex}
      style={{
        position: "relative",
        background: "#faf9f6",
        border: "1px solid #e0ddd9",
        borderRadius: "10px",
        padding: "10px 14px",
        width: "200px",
        boxShadow: "0 1px 0 rgba(12,26,46,0.04), 0 1px 2px rgba(12,26,46,0.04)",
        fontFamily:
          'Inter, system-ui, -apple-system, "Segoe UI", sans-serif',
      }}
    >
      {hasError && (
        <div
          data-testid={DATA_TEST_ID.VALIDATION_BADGE}
          style={{
            position: "absolute",
            top: "-8px",
            right: "-8px",
          }}
        >
          <ValidationBadge severity="error" count={1} title="Validation error" />
        </div>
      )}

      <Handle
        type="target"
        position={Position.Left}
        data-testid={DATA_TEST_ID.INPUT_HANDLE}
        style={{
          background: inputColor,
          border: "2px solid #faf9f6",
          width: "10px",
          height: "10px",
        }}
      />

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          marginBottom: configPreview ? "6px" : 0,
        }}
      >
        <Icon name="config" size={14} style={{ color: "#6b6b6b", flexShrink: 0 }} />

        <span
          data-testid={DATA_TEST_ID.PLUGIN_NAME}
          style={{
            fontSize: "13px",
            fontWeight: 500,
            color: "#0c1a2e",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            flex: 1,
          }}
        >
          {pluginName}
        </span>

        <span
          data-testid={DATA_TEST_ID.STATUS_INDICATOR}
          title={status ?? "unknown"}
          style={{
            width: "6px",
            height: "6px",
            borderRadius: "50%",
            backgroundColor: statusColor,
            flexShrink: 0,
          }}
        />
      </div>

      {configPreview && (
        <div
          data-testid={DATA_TEST_ID.CONFIG_PREVIEW}
          style={{
            fontSize: "11px",
            color: "#6b6b6b",
            fontFamily:
              '"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace',
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            marginBottom: "6px",
          }}
        >
          {configPreview}
        </div>
      )}

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: "6px",
        }}
      >
        <span
          data-testid={DATA_TEST_ID.INPUT_SLOT_PILL}
          style={{
            fontSize: "10px",
            padding: "1px 6px",
            borderRadius: "10px",
            backgroundColor: `${inputColor}22`,
            color: inputColor,
            border: `1px solid ${inputColor}44`,
            fontWeight: 500,
          }}
        >
          {inputSlotType}
        </span>

        <span
          data-testid={DATA_TEST_ID.OUTPUT_SLOT_PILL}
          style={{
            fontSize: "10px",
            padding: "1px 6px",
            borderRadius: "10px",
            backgroundColor: `${outputColor}22`,
            color: outputColor,
            border: `1px solid ${outputColor}44`,
            fontWeight: 500,
          }}
        >
          {outputSlotType}
        </span>
      </div>

      <Handle
        type="source"
        position={Position.Right}
        data-testid={DATA_TEST_ID.OUTPUT_HANDLE}
        style={{
          background: outputColor,
          border: "2px solid #faf9f6",
          width: "10px",
          height: "10px",
        }}
      />
    </div>
  );
}
