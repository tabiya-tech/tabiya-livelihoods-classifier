import type { PluginCategory, PluginSummary } from "@/lib/api";
import { usePluginCatalog } from "../../hooks/usePluginCatalog";

const uniqueId = "e1f2a3b4-c5d6-e7f8-a9b0-c1d2e3f4a5b6";

export const DATA_TEST_ID = {
  ROOT: `plugin-palette-root-${uniqueId}`,
  LOADING: `plugin-palette-loading-${uniqueId}`,
  ERROR: `plugin-palette-error-${uniqueId}`,
  CATEGORY_SECTION: `plugin-palette-category-section-${uniqueId}`,
  CATEGORY_BADGE: `plugin-palette-category-badge-${uniqueId}`,
  PLUGIN_ROW: `plugin-palette-plugin-row-${uniqueId}`,
  COMING_SOON_PILL: `plugin-palette-coming-soon-pill-${uniqueId}`,
};

const CATEGORY_ORDER: Array<PluginCategory | "other"> = [
  "source",
  "core",
  "transform",
  "sink",
  "other",
];

const CATEGORY_LABELS: Record<PluginCategory | "other", string> = {
  source: "Source",
  core: "Core",
  transform: "Transform",
  sink: "Sink",
  other: "Other",
};

const CATEGORY_COLORS: Record<PluginCategory | "other", string> = {
  source: "#d4ebe6",
  core: "#d9e3f0",
  transform: "#f3ecc7",
  sink: "#f4d8dc",
  other: "#ece9e3",
};

const CATEGORY_TEXT_COLORS: Record<PluginCategory | "other", string> = {
  source: "#26887d",
  core: "#002147",
  transform: "#b8860b",
  sink: "#b03a4a",
  other: "#6b6b6b",
};

function groupPluginsByCategory(
  plugins: PluginSummary[],
): Map<PluginCategory | "other", PluginSummary[]> {
  const groups = new Map<PluginCategory | "other", PluginSummary[]>();

  for (const plugin of plugins) {
    const category: PluginCategory | "other" = plugin.category ?? "other";
    const existing = groups.get(category) ?? [];
    groups.set(category, [...existing, plugin]);
  }

  return groups;
}

export interface PluginPaletteProps {
  className?: string;
  /**
   * Optional click handler. When set, non-`coming_soon` plugin rows behave
   * like buttons and fire this callback with the plugin_id. Existing drag
   * behavior stays intact.
   */
  onPluginClick?: (pluginId: string) => void;
}

export function PluginPalette({
  className,
  onPluginClick,
}: PluginPaletteProps) {
  const { status, plugins, error } = usePluginCatalog();

  if (status === "loading") {
    return (
      <div
        data-testid={DATA_TEST_ID.LOADING}
        style={{
          padding: "16px",
          color: "#6b6b6b",
          fontSize: "13px",
          fontFamily: 'Inter, system-ui, -apple-system, "Segoe UI", sans-serif',
        }}
      >
        Loading plugins...
      </div>
    );
  }

  if (status === "error") {
    return (
      <div
        data-testid={DATA_TEST_ID.ERROR}
        style={{
          padding: "16px",
          color: "#c0392b",
          fontSize: "13px",
          fontFamily: 'Inter, system-ui, -apple-system, "Segoe UI", sans-serif',
        }}
      >
        {error?.message ?? "Failed to load plugins."}
      </div>
    );
  }

  const groupedPlugins = groupPluginsByCategory(plugins);

  return (
    <div
      data-testid={DATA_TEST_ID.ROOT}
      className={className}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "16px",
        padding: "12px",
        fontFamily: 'Inter, system-ui, -apple-system, "Segoe UI", sans-serif',
      }}
    >
      {CATEGORY_ORDER.filter((category) => groupedPlugins.has(category)).map(
        (category) => {
          const categoryPlugins = groupedPlugins.get(category) ?? [];
          const backgroundColor = CATEGORY_COLORS[category];
          const textColor = CATEGORY_TEXT_COLORS[category];
          const label = CATEGORY_LABELS[category];

          return (
            <section
              key={category}
              data-testid={DATA_TEST_ID.CATEGORY_SECTION}
              data-category={category}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  marginBottom: "8px",
                }}
              >
                <span
                  data-testid={DATA_TEST_ID.CATEGORY_BADGE}
                  style={{
                    fontSize: "11px",
                    fontWeight: 600,
                    padding: "2px 8px",
                    borderRadius: "10px",
                    backgroundColor,
                    color: textColor,
                    letterSpacing: "0.03em",
                    textTransform: "uppercase",
                  }}
                >
                  {label}
                </span>
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "6px",
                }}
              >
                {categoryPlugins.map((plugin) => (
                  <PluginRow
                    key={plugin.plugin_id}
                    plugin={plugin}
                    onClick={onPluginClick}
                  />
                ))}
              </div>
            </section>
          );
        },
      )}
    </div>
  );
}

interface PluginRowProps {
  plugin: PluginSummary;
  onClick?: (pluginId: string) => void;
}

function PluginRow({ plugin, onClick }: PluginRowProps) {
  const isDisabled = plugin.coming_soon;
  const isClickable = !isDisabled && Boolean(onClick);

  function handleDragStart(event: React.DragEvent<HTMLDivElement>) {
    event.dataTransfer.setData(
      "application/pipeline-plugin",
      JSON.stringify({ pluginId: plugin.plugin_id, pluginSummary: plugin }),
    );
  }

  function handleClick() {
    if (isDisabled || !onClick) return;
    onClick(plugin.plugin_id);
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    if (!isClickable) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onClick?.(plugin.plugin_id);
    }
  }

  return (
    <div
      data-testid={DATA_TEST_ID.PLUGIN_ROW}
      data-plugin-id={plugin.plugin_id}
      draggable={!isDisabled}
      onDragStart={isDisabled ? undefined : handleDragStart}
      onClick={isClickable ? handleClick : undefined}
      onKeyDown={isClickable ? handleKeyDown : undefined}
      role={isClickable ? "button" : undefined}
      tabIndex={isClickable ? 0 : undefined}
      style={{
        padding: "8px",
        borderRadius: "6px",
        border: "1px solid #e0ddd9",
        backgroundColor: isDisabled ? "#f3f1ee" : "#faf9f6",
        opacity: isDisabled ? 0.6 : 1,
        cursor: isDisabled ? "default" : isClickable ? "pointer" : "grab",
        display: "flex",
        flexDirection: "column",
        gap: "2px",
        boxShadow: "0 1px 0 rgba(12,26,46,0.04)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "4px",
        }}
      >
        <span
          style={{
            fontSize: "12px",
            fontWeight: 500,
            color: isDisabled ? "#8a8780" : "#0c1a2e",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            flex: 1,
          }}
        >
          {plugin.name}
        </span>

        {isDisabled && (
          <span
            data-testid={DATA_TEST_ID.COMING_SOON_PILL}
            style={{
              fontSize: "9px",
              fontWeight: 600,
              padding: "1px 5px",
              borderRadius: "8px",
              backgroundColor: "#e0ddd9",
              color: "#6b6b6b",
              letterSpacing: "0.03em",
              textTransform: "uppercase",
              flexShrink: 0,
            }}
          >
            Soon
          </span>
        )}
      </div>

      <span
        style={{
          fontSize: "11px",
          color: "#8a8780",
          overflow: "hidden",
          textOverflow: "ellipsis",
          whiteSpace: "nowrap",
          lineHeight: 1.3,
        }}
      >
        {plugin.summary}
      </span>
    </div>
  );
}
