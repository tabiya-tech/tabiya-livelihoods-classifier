import type { Meta, StoryObj } from "@storybook/react";
import { fixturePluginSummaries } from "@/mocks/fixtures/plugins";
import { PluginCatalogOverrideContext } from "../../hooks/usePluginCatalog";
import { PluginPalette } from "./PluginPalette";

function PluginPaletteHarness({
  status,
  plugins,
  error,
}: {
  status: "loading" | "ready" | "error";
  plugins: typeof fixturePluginSummaries;
  error: Error | null;
}) {
  const snapshot = {
    status,
    plugins,
    error,
    refetch: async () => {},
  };

  return (
    <PluginCatalogOverrideContext.Provider value={snapshot}>
      <div style={{ width: "280px", border: "1px solid #e0ddd9", borderRadius: "10px", overflow: "auto" }}>
        <PluginPalette />
      </div>
    </PluginCatalogOverrideContext.Provider>
  );
}

const meta: Meta<typeof PluginPaletteHarness> = {
  title: "Features/Pipelines/PluginPalette",
  component: PluginPaletteHarness,
  parameters: { layout: "centered" },
};
export default meta;

type Story = StoryObj<typeof PluginPaletteHarness>;

export const Default: Story = {
  args: {
    status: "ready",
    plugins: fixturePluginSummaries,
    error: null,
  },
};

export const Loading: Story = {
  args: {
    status: "loading",
    plugins: [],
    error: null,
  },
};

export const ErrorState: Story = {
  args: {
    status: "error",
    plugins: [],
    error: new Error("Network request failed"),
  },
};
