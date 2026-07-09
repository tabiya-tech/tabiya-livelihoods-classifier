import type { Meta, StoryObj } from "@storybook/react";
import { BrowserRouter } from "react-router-dom";
import type { PluginDetail } from "@/lib/api";
import { fixtureNerManifest } from "@/mocks/fixtures/plugins";
import {
  PluginDetailOverrideContext,
  type PluginDetailSnapshot,
} from "../hooks/usePluginDetail";
import { PipelineLibraryPage } from "./PipelineLibraryPage";

const nerDetail: PluginDetail = {
  plugin_id: fixtureNerManifest.plugin_id,
  status: "enabled",
  coming_soon: false,
  last_error: null,
  manifest: fixtureNerManifest,
};

const readyNerSnapshot: PluginDetailSnapshot = {
  status: "ready",
  detail: nerDetail,
  error: null,
};

interface LibraryHarnessProps {
  detailSnapshot: PluginDetailSnapshot | null;
}

function LibraryHarness({ detailSnapshot }: LibraryHarnessProps) {
  return (
    <BrowserRouter>
      <PluginDetailOverrideContext.Provider value={detailSnapshot}>
        <PipelineLibraryPage />
      </PluginDetailOverrideContext.Provider>
    </BrowserRouter>
  );
}

const meta: Meta<typeof LibraryHarness> = {
  title: "Features/Pipelines/PipelineLibraryPage",
  component: LibraryHarness,
  parameters: { layout: "fullscreen" },
};
export default meta;

type Story = StoryObj<typeof LibraryHarness>;

export const Default: Story = {
  args: {
    detailSnapshot: null,
  },
};

export const WithSelectedPlugin: Story = {
  args: {
    detailSnapshot: readyNerSnapshot,
  },
};
