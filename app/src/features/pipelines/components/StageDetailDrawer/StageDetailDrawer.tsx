/**
 * Drawer panel for configuring a single pipeline stage.
 *
 * Shows the plugin name as the drawer title, slot type pills as the
 * description, an error block when validation issues are present, and a
 * ConfigForm wired to the stage's config. A delete button and a close
 * button live in the footer.
 */

import { Button, Drawer, Tag } from "@/components";
import type { PipelineStage, PluginManifest, PipelineValidationIssue } from "@/lib/api";
import { ConfigForm } from "../ConfigForm/ConfigForm";

const uniqueId = "f9e8d7c6-b5a4-4321-8fed-cba987654321";

export const DATA_TEST_ID = {
  ROOT: `stage-detail-drawer-root-${uniqueId}`,
  SLOT_PILLS: `stage-detail-drawer-slot-pills-${uniqueId}`,
  ERRORS_BLOCK: `stage-detail-drawer-errors-block-${uniqueId}`,
  DELETE_BUTTON: `stage-detail-drawer-delete-button-${uniqueId}`,
  CLOSE_BUTTON: `stage-detail-drawer-close-button-${uniqueId}`,
};

export interface StageDetailDrawerProps {
  open: boolean;
  onClose: () => void;
  stage?: PipelineStage;
  stageIndex?: number;
  manifest?: PluginManifest;
  /** Validation issues scoped to this stage. */
  errors?: PipelineValidationIssue[];
  onChange: (nextStage: PipelineStage) => void;
  onDelete: () => void;
}

export function StageDetailDrawer({
  open,
  onClose,
  stage,
  stageIndex,
  manifest,
  errors,
  onChange,
  onDelete,
}: StageDetailDrawerProps) {
  const eyebrowLabel =
    stageIndex !== undefined ? `Stage ${stageIndex + 1}` : undefined;

  const drawerTitle = manifest?.name ?? stage?.plugin_id ?? "Stage";

  const slotPillsNode =
    manifest ? (
      <span
        data-testid={DATA_TEST_ID.SLOT_PILLS}
        style={{ display: "inline-flex", gap: "6px" }}
      >
        <Tag tone="neutral" size="sm">
          {`In: ${manifest.input_slot.type}`}
        </Tag>
        <Tag tone="neutral" size="sm">
          {`Out: ${manifest.output_slot.type}`}
        </Tag>
      </span>
    ) : undefined;

  const hasErrors = errors && errors.length > 0;

  const footerNode = (
    <span style={{ display: "flex", gap: "8px", justifyContent: "flex-end" }}>
      <Button
        variant="danger"
        data-testid={DATA_TEST_ID.DELETE_BUTTON}
        onClick={onDelete}
      >
        Delete stage
      </Button>
      <Button
        variant="ghost"
        data-testid={DATA_TEST_ID.CLOSE_BUTTON}
        onClick={onClose}
      >
        Close
      </Button>
    </span>
  );

  return (
    <div data-testid={DATA_TEST_ID.ROOT}>
      <Drawer
        open={open}
        onClose={onClose}
        eyebrow={eyebrowLabel}
        title={drawerTitle}
        description={slotPillsNode}
        footer={footerNode}
      >
        {hasErrors && (
          <div
            data-testid={DATA_TEST_ID.ERRORS_BLOCK}
            style={{
              border: "1px solid #ef4444",
              borderRadius: "6px",
              padding: "10px 14px",
              marginBottom: "16px",
              backgroundColor: "#fef2f2",
            }}
          >
            <p
              style={{
                fontSize: "12px",
                fontWeight: 600,
                color: "#b91c1c",
                marginBottom: "6px",
              }}
            >
              Validation errors
            </p>
            <ul style={{ paddingLeft: "16px", margin: 0 }}>
              {errors!.map((issue, issueIndex) => (
                <li
                  key={`${issue.code}-${issueIndex}`}
                  style={{ fontSize: "12px", color: "#b91c1c" }}
                >
                  {issue.message}
                </li>
              ))}
            </ul>
          </div>
        )}

        {stage ? (
          <ConfigForm
            schema={manifest?.config_schema ?? { type: "object", properties: {} }}
            value={stage.config}
            onChange={(nextConfig) => onChange({ ...stage, config: nextConfig })}
            errors={errors?.map((issue) => ({
              path: issue.detail ? Object.keys(issue.detail) : [],
              message: issue.message,
            }))}
            pluginId={stage.plugin_id}
          />
        ) : (
          <p style={{ fontSize: "13px", color: "#6b6b6b" }}>
            Select a stage to configure it.
          </p>
        )}
      </Drawer>
    </div>
  );
}
