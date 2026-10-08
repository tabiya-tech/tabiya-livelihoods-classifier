/**
 * Full pipelines feature story — the whole flow wired together against the
 * MSW mock backend, so you can interact end-to-end: browse the list, open the
 * editor, drag plugins from the palette, configure stages, save, activate,
 * clone, delete, and visit the library.
 *
 * Unlike the per-page stories (which stub individual hooks via override
 * providers), this mounts the REAL pages on a MemoryRouter and lets the real
 * hooks talk to MSW. It's the closest Storybook gets to the running app.
 */

import type { Meta, StoryObj } from "@storybook/react";
import { MemoryRouter, Route, Routes, Link } from "react-router-dom";
import { ToastProvider } from "@/components";
import { NavigationGuardProvider } from "@/lib/navigationGuard";
import { routerPaths } from "@/routes/routerPaths";
import {
  seedPipelinesHandlersStore,
  resetPipelinesHandlersStore,
} from "@/mocks/handlers";
import { fixturePipelines } from "@/mocks/fixtures/pipelines";
import { PipelinesPage } from "./PipelinesPage/PipelinesPage";
import { PipelineEditorPage } from "./PipelineEditorPage/PipelineEditorPage";
import { PipelineLibraryPage } from "./PipelineLibraryPage/PipelineLibraryPage";

/** Minimal nav bar so you can move between the feature's routes in Storybook. */
function FeatureNav() {
  return (
    <nav
      style={{
        display: "flex",
        gap: "16px",
        padding: "10px 24px",
        borderBottom: "1px solid #e0ddd9",
        fontFamily: 'Inter, system-ui, sans-serif',
        fontSize: "13px",
      }}
    >
      <Link to={routerPaths.PIPELINES}>Pipelines</Link>
      <Link to={routerPaths.PIPELINE_NEW}>New</Link>
      <Link to={routerPaths.PIPELINE_LIBRARY}>Library</Link>
    </nav>
  );
}

function PipelinesFeature({ initialPath }: { initialPath: string }) {
  return (
    <MemoryRouter initialEntries={[initialPath]}>
      <NavigationGuardProvider>
        <ToastProvider>
          <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
            <FeatureNav />
            <div style={{ flex: 1, minHeight: 0 }}>
              <Routes>
                <Route path={routerPaths.PIPELINES} element={<PipelinesPage />} />
                <Route path={routerPaths.PIPELINE_NEW} element={<PipelineEditorPage />} />
                <Route
                  path={routerPaths.PIPELINE_LIBRARY}
                  element={<PipelineLibraryPage />}
                />
                <Route
                  path={routerPaths.PIPELINE_EDIT}
                  element={<PipelineEditorPage />}
                />
              </Routes>
            </div>
          </div>
        </ToastProvider>
      </NavigationGuardProvider>
    </MemoryRouter>
  );
}

const meta: Meta<typeof PipelinesFeature> = {
  title: "Features/Pipelines/Full Feature",
  component: PipelinesFeature,
  parameters: { layout: "fullscreen" },
};
export default meta;

type Story = StoryObj<typeof PipelinesFeature>;

/** Start on the list, seeded with the fixture pipelines. */
export const List: Story = {
  args: { initialPath: routerPaths.PIPELINES },
  beforeEach() {
    seedPipelinesHandlersStore(fixturePipelines);
  },
};

/** Start on the empty editor to build a pipeline from scratch. */
export const NewPipeline: Story = {
  args: { initialPath: routerPaths.PIPELINE_NEW },
  beforeEach() {
    resetPipelinesHandlersStore();
  },
};

/** Start on the plugin library. */
export const Library: Story = {
  args: { initialPath: routerPaths.PIPELINE_LIBRARY },
  beforeEach() {
    resetPipelinesHandlersStore();
  },
};
