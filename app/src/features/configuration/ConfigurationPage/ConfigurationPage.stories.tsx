import { useState } from "react";
import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { MemoryRouter } from "react-router-dom";
import { ToastProvider } from "@/components";
import { NavigationGuardProvider } from "@/lib/navigationGuard";
import type { V2UserConfig } from "@/lib/api";
import {
  fixtureNelModels,
  fixtureTaxonomyModels,
  fixtureUserConfig,
} from "@/mocks/fixtures/nelV2";
import { ConfigurationOverridesProvider } from "../hooks/configurationOverrides";
import type {
  NelModelsSnapshot,
} from "../hooks/useNelModels";
import type {
  TaxonomyModelsSnapshot,
} from "../hooks/useTaxonomyModels";
import type {
  UserConfigurationState,
  ConfigurationSaveStatus,
} from "../hooks/useUserConfiguration";
import { ConfigurationPage } from "./ConfigurationPage";

const readyNelModels: NelModelsSnapshot = {
  status: "ready",
  models: fixtureNelModels,
  error: null,
};

const readyTaxonomyModels: TaxonomyModelsSnapshot = {
  status: "ready",
  models: fixtureTaxonomyModels,
  error: null,
};

interface UserConfigurationHarnessProps {
  /** Seed selection that the story renders as "persisted". */
  seedConfig: V2UserConfig;
  /** Optional draft override — defaults to the seed. */
  initialDraft?: V2UserConfig;
  /** Save status the story locks the bar into. */
  saveStatus: ConfigurationSaveStatus;
  /** Optional save behavior; defaults to resolving immediately. */
  onSave?: (config: V2UserConfig) => Promise<V2UserConfig>;
  children: React.ReactNode;
}

/**
 * Wraps the page in a stateful override that mimics the real hook so the
 * story can be interacted with end-to-end (click a different model → bar
 * appears; discard → bar disappears) without touching the real backend.
 */
function UserConfigurationHarness({
  seedConfig,
  initialDraft,
  saveStatus,
  onSave,
  children,
}: UserConfigurationHarnessProps) {
  const [saved, setSaved] = useState<V2UserConfig>(seedConfig);
  const [draft, setDraft] = useState<V2UserConfig>(
    initialDraft ?? seedConfig,
  );
  const [currentSaveStatus, setCurrentSaveStatus] =
    useState<ConfigurationSaveStatus>(saveStatus);
  const [savedAt, setSavedAt] = useState<number | null>(null);

  const isDirty =
    saved.nel_model_id !== draft.nel_model_id ||
    saved.taxonomy_model_id !== draft.taxonomy_model_id;

  const value: UserConfigurationState = {
    loadStatus: "ready",
    loadError: null,
    saveStatus: currentSaveStatus,
    saveError: null,
    saved,
    draft,
    isDirty,
    savedAt,
    setDraft: (patch) => {
      setDraft((previous) => ({ ...previous, ...patch }));
      if (currentSaveStatus === "saved") setCurrentSaveStatus("idle");
    },
    discard: () => {
      setDraft(saved);
      setCurrentSaveStatus("idle");
    },
    save: async () => {
      setCurrentSaveStatus("saving");
      if (onSave) {
        const persisted = await onSave(draft);
        setSaved(persisted);
        setDraft(persisted);
      } else {
        setSaved(draft);
      }
      setSavedAt(Date.now());
      setCurrentSaveStatus("saved");
    },
  };

  return (
    <ConfigurationOverridesProvider
      nelModels={readyNelModels}
      taxonomyModels={readyTaxonomyModels}
      userConfiguration={value}
    >
      {children}
    </ConfigurationOverridesProvider>
  );
}

const meta: Meta<typeof ConfigurationPage> = {
  title: "Features/Configuration/ConfigurationPage",
  component: ConfigurationPage,
  parameters: { layout: "fullscreen" },
  decorators: [
    function StoryWithProviders(StoryComponent) {
      return (
        <MemoryRouter initialEntries={["/configuration"]}>
          <ToastProvider>
            <NavigationGuardProvider>
              <StoryComponent />
            </NavigationGuardProvider>
          </ToastProvider>
        </MemoryRouter>
      );
    },
  ],
};
export default meta;

type Story = StoryObj<typeof ConfigurationPage>;

/**
 * The default landing state: persisted config matches the draft, nothing
 * dirty, save bar hidden.
 */
export const Initial: Story = {
  render: function InitialStory() {
    return (
      <UserConfigurationHarness
        seedConfig={fixtureUserConfig}
        saveStatus="idle"
      >
        <ConfigurationPage />
      </UserConfigurationHarness>
    );
  },
};

/**
 * The user has switched the NEL model — draft differs from saved, the save
 * bar slides up with the unsaved-changes copy.
 */
export const Dirty: Story = {
  render: function DirtyStory() {
    const dirtyDraft: V2UserConfig = {
      ...fixtureUserConfig,
      nel_model_id: fixtureNelModels[0].model_id,
    };
    return (
      <UserConfigurationHarness
        seedConfig={fixtureUserConfig}
        initialDraft={dirtyDraft}
        saveStatus="idle"
      >
        <ConfigurationPage />
      </UserConfigurationHarness>
    );
  },
};

/**
 * The user has clicked Save and the PUT is in flight. Save bar shows the
 * loading copy and the Discard button is disabled.
 */
export const Saving: Story = {
  render: function SavingStory() {
    const dirtyDraft: V2UserConfig = {
      ...fixtureUserConfig,
      nel_model_id: fixtureNelModels[0].model_id,
    };
    return (
      <UserConfigurationHarness
        seedConfig={fixtureUserConfig}
        initialDraft={dirtyDraft}
        saveStatus="saving"
      >
        <ConfigurationPage />
      </UserConfigurationHarness>
    );
  },
};

/**
 * The PUT failed. Save bar stays in the dirty state and the toast surfaces
 * the error.
 */
export const SaveError: Story = {
  render: function SaveErrorStory() {
    const dirtyDraft: V2UserConfig = {
      ...fixtureUserConfig,
      nel_model_id: fixtureNelModels[0].model_id,
    };
    const onSave = fn(async () => {
      throw new Error("Backend unreachable");
    });
    return (
      <UserConfigurationHarness
        seedConfig={fixtureUserConfig}
        initialDraft={dirtyDraft}
        saveStatus="idle"
        onSave={onSave}
      >
        <ConfigurationPage />
      </UserConfigurationHarness>
    );
  },
};
