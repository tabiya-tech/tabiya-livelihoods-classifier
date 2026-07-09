/**
 * Documentation-only browse view for the plugin catalog. Mounted at
 * `/pipelines/library`. Two-column layout:
 *
 *   ┌────────────────────────┬─────────────────────────────────────┐
 *   │  PluginPalette (250px) │  Detail panel (flex-1)              │
 *   └────────────────────────┴─────────────────────────────────────┘
 *
 * The palette is reused with its optional `onPluginClick`; clicking a
 * non-`coming_soon` row selects the plugin and populates the detail panel
 * on the right with its full manifest (slots, config schema, capabilities,
 * timeout). No mutations happen here — this page never writes.
 */

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { EmptyState, Spinner, Tag } from "@/components";
import type { PluginManifest, PluginSlot } from "@/lib/api";
import { PluginPalette } from "../components/PluginPalette/PluginPalette";
import { usePluginDetail } from "../hooks/usePluginDetail";

const uniqueId = "e6f7a8b9-c0d1-4e2f-a3b4-5c6d7e8f9a0b";

export const DATA_TEST_ID = {
  CONTAINER: `pipeline-library-page-container-${uniqueId}`,
  DETAIL_PANEL: `pipeline-library-page-detail-panel-${uniqueId}`,
  DETAIL_EMPTY_PROMPT: `pipeline-library-page-detail-empty-prompt-${uniqueId}`,
  DETAIL_FIELD: `pipeline-library-page-detail-field-${uniqueId}`,
  DETAIL_LOADING: `pipeline-library-page-detail-loading-${uniqueId}`,
  DETAIL_ERROR: `pipeline-library-page-detail-error-${uniqueId}`,
  DETAIL_UNAVAILABLE: `pipeline-library-page-detail-unavailable-${uniqueId}`,
};

interface SchemaField {
  name: string;
  type: string;
  title?: string;
}

interface SchemaLike {
  properties?: Record<string, unknown>;
}

function extractSchemaFields(schema: Record<string, unknown>): SchemaField[] {
  const properties = (schema as SchemaLike).properties;
  if (!properties || typeof properties !== "object") return [];

  return Object.entries(properties).map(([fieldName, rawFieldSchema]) => {
    const fieldSchema = (rawFieldSchema ?? {}) as Record<string, unknown>;
    const rawType = fieldSchema.type;
    const type = typeof rawType === "string" ? rawType : "unknown";
    const rawTitle = fieldSchema.title;
    const title = typeof rawTitle === "string" ? rawTitle : undefined;
    return { name: fieldName, type, title };
  });
}

interface CapabilityEntry {
  key: string;
  value: string;
}

const CAPABILITY_KEYS = [
  "x-tabiya-contract-version",
  "x-tabiya-streams",
  "x-tabiya-idempotent",
  "x-tabiya-cancellable",
  "x-tabiya-batch-max",
] as const;

function extractCapabilities(manifest: PluginManifest): CapabilityEntry[] {
  const capabilities: CapabilityEntry[] = [];
  for (const key of CAPABILITY_KEYS) {
    const rawValue = manifest[key];
    if (rawValue === undefined || rawValue === null) continue;
    capabilities.push({ key, value: String(rawValue) });
  }
  return capabilities;
}

function formatSlot(slot: PluginSlot): string {
  if (slot.cardinality && slot.cardinality !== "single") {
    return `${slot.type} (${slot.cardinality})`;
  }
  return slot.type;
}

export function PipelineLibraryPage() {
  const { t } = useTranslation();
  const [selectedPluginId, setSelectedPluginId] = useState<string>("");
  const detailSnapshot = usePluginDetail(selectedPluginId);

  function handlePluginClick(pluginId: string) {
    setSelectedPluginId(pluginId);
  }

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className="flex h-full flex-col overflow-hidden"
    >
      <header className="flex flex-col gap-2 border-b border-line px-8 py-6">
        <span className="eyebrow">{t("pipelines.library.eyebrow")}</span>
        <h1 className="h-page m-0">{t("pipelines.library.title")}</h1>
        <p className="m-0 max-w-[680px] text-sm leading-relaxed text-muted">
          {t("pipelines.library.intro")}
        </p>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <aside className="w-[250px] shrink-0 overflow-y-auto border-r border-line bg-paper">
          <PluginPalette onPluginClick={handlePluginClick} />
        </aside>

        <main
          data-testid={DATA_TEST_ID.DETAIL_PANEL}
          className="flex-1 overflow-y-auto px-8 py-6"
        >
          <PluginDetailContent
            selectedPluginId={selectedPluginId}
            snapshotStatus={detailSnapshot.status}
            manifest={detailSnapshot.detail?.manifest ?? null}
            emptyPrompt={t("pipelines.library.detail.emptyPrompt")}
          />
        </main>
      </div>
    </div>
  );
}

interface PluginDetailContentProps {
  selectedPluginId: string;
  snapshotStatus: "idle" | "loading" | "ready" | "error";
  manifest: PluginManifest | null;
  emptyPrompt: string;
}

function PluginDetailContent({
  selectedPluginId,
  snapshotStatus,
  manifest,
  emptyPrompt,
}: PluginDetailContentProps) {
  const { t } = useTranslation();

  if (!selectedPluginId || snapshotStatus === "idle") {
    return (
      <div
        data-testid={DATA_TEST_ID.DETAIL_EMPTY_PROMPT}
        className="flex h-full items-center justify-center"
      >
        <EmptyState icon="pipelines" title={emptyPrompt} />
      </div>
    );
  }

  if (snapshotStatus === "loading") {
    return (
      <div
        data-testid={DATA_TEST_ID.DETAIL_LOADING}
        className="flex items-center gap-2 text-sm text-muted"
      >
        <Spinner />
        {t("common.loading")}
      </div>
    );
  }

  if (snapshotStatus === "error" || !manifest) {
    return (
      <div
        data-testid={
          snapshotStatus === "error"
            ? DATA_TEST_ID.DETAIL_ERROR
            : DATA_TEST_ID.DETAIL_UNAVAILABLE
        }
        className="text-sm text-error"
      >
        <EmptyState icon="close" title={emptyPrompt} />
      </div>
    );
  }

  const schemaFields = extractSchemaFields(manifest.config_schema);
  const capabilities = extractCapabilities(manifest);
  const inputSlot = formatSlot(manifest.input_slot);
  const outputSlot = formatSlot(manifest.output_slot);

  return (
    <article className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <div className="flex items-baseline gap-3">
          <h2 className="m-0 text-lg font-semibold text-navy">
            {manifest.name}
          </h2>
          <span className="font-mono text-xs text-muted">
            v{manifest.version}
          </span>
        </div>
        {manifest.summary && (
          <p className="m-0 text-sm text-muted">{manifest.summary}</p>
        )}
      </header>

      <section className="flex flex-col gap-2">
        <h3 className="m-0 text-xs font-semibold uppercase tracking-wide text-muted">
          {t("pipelines.library.detail.slotsHeading")}
        </h3>
        <div className="flex items-center gap-2">
          <Tag tone="neutral" size="sm">
            {t("pipelines.library.detail.slotInput")}: {inputSlot}
          </Tag>
          <Tag tone="neutral" size="sm">
            {t("pipelines.library.detail.slotOutput")}: {outputSlot}
          </Tag>
        </div>
      </section>

      <section className="flex flex-col gap-2">
        <h3 className="m-0 text-xs font-semibold uppercase tracking-wide text-muted">
          {t("pipelines.library.detail.schemaHeading")}
        </h3>
        {schemaFields.length === 0 ? (
          <p className="m-0 text-sm text-muted">—</p>
        ) : (
          <ul className="m-0 flex flex-col gap-1 pl-0">
            {schemaFields.map((field) => (
              <li
                key={field.name}
                data-testid={DATA_TEST_ID.DETAIL_FIELD}
                data-field-name={field.name}
                className="flex items-baseline gap-2 rounded border border-line bg-paper px-3 py-2 text-sm"
              >
                <code className="font-mono text-xs text-navy">
                  {field.name}
                </code>
                <span className="font-mono text-xs text-muted">
                  {field.type}
                </span>
                {field.title && (
                  <span className="ml-auto text-xs text-muted">
                    {field.title}
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>

      {capabilities.length > 0 && (
        <section className="flex flex-col gap-2">
          <h3 className="m-0 text-xs font-semibold uppercase tracking-wide text-muted">
            {t("pipelines.library.detail.capabilitiesHeading")}
          </h3>
          <ul className="m-0 flex flex-wrap gap-2 pl-0">
            {capabilities.map((capability) => (
              <li key={capability.key} className="list-none">
                <Tag tone="neutral" size="sm">
                  {capability.key}: {capability.value}
                </Tag>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="flex flex-col gap-1 border-t border-line pt-4 font-mono text-xs text-muted">
        <span>
          {t("pipelines.library.detail.timeoutLabel")}: {manifest.timeout_ms}ms
        </span>
      </section>
    </article>
  );
}
