import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import {
  fixtureNerManifest,
  fixtureNelManifest,
  fixturePluginOptions,
} from "@/mocks/fixtures/plugins";
import type { PluginOptionsState } from "../../hooks/usePluginOptions";
import { PluginOptionsOverrideContext } from "../../hooks/pipelinesOverrides";
import { ConfigForm } from "./ConfigForm";

const meta: Meta<typeof ConfigForm> = {
  title: "Features/Pipelines/ConfigForm",
  component: ConfigForm,
  parameters: { layout: "centered" },
  args: {
    onChange: fn(),
  },
  decorators: [
    function ContainerDecorator(Story) {
      return (
        <div style={{ width: 420, padding: 24 }}>
          <Story />
        </div>
      );
    },
  ],
};
export default meta;

type Story = StoryObj<typeof ConfigForm>;

export const StringField: Story = {
  args: {
    pluginId: "tabiya.test.v1",
    schema: {
      type: "object",
      properties: {
        description: {
          type: "string",
          title: "Description",
          description: "A free-form text description.",
        },
      },
    },
    value: { description: "Sample description text" },
  },
};

export const StringEnum: Story = {
  args: {
    pluginId: "tabiya.test.v1",
    schema: {
      type: "object",
      properties: {
        language: {
          type: "string",
          title: "Language",
          enum: ["English", "French", "German", "Spanish"],
        },
      },
    },
    value: { language: "English" },
  },
};

export const IntegerSlider: Story = {
  args: {
    pluginId: fixtureNelManifest.plugin_id,
    schema: {
      type: "object",
      properties: {
        top_k: {
          type: "integer",
          title: "Top K",
          minimum: 1,
          maximum: 50,
          default: 5,
        },
      },
    },
    value: { top_k: 5 },
  },
};

export const BooleanToggle: Story = {
  args: {
    pluginId: "tabiya.test.v1",
    schema: {
      type: "object",
      properties: {
        verbose: {
          type: "boolean",
          title: "Verbose mode",
          description: "Enable verbose logging.",
        },
      },
    },
    value: { verbose: false },
  },
};

export const ArrayEnum: Story = {
  args: {
    pluginId: fixtureNerManifest.plugin_id,
    schema: {
      type: "object",
      properties: {
        entity_types: {
          type: "array",
          title: "Entity types",
          items: {
            type: "string",
            enum: ["occupation", "skill", "qualification", "experience", "domain"],
          },
        },
      },
    },
    value: { entity_types: ["occupation", "skill"] },
  },
};

export const XSourceField: Story = {
  args: {
    pluginId: fixtureNerManifest.plugin_id,
    schema: {
      type: "object",
      properties: {
        model_id: {
          type: "string",
          title: "Model",
          "x-source": "/v2/plugins/tabiya.ner.v1/options/model_id",
        },
      },
    },
    value: { model_id: "tabiya/roberta-base-job-ner" },
  },
  decorators: [
    function PluginOptionsDecorator(Story) {
      const readyState: PluginOptionsState = {
        status: "ready",
        options: fixturePluginOptions["tabiya.ner.v1"].model_id.options,
        error: null,
      };
      return (
        <PluginOptionsOverrideContext.Provider value={readyState}>
          <Story />
        </PluginOptionsOverrideContext.Provider>
      );
    },
  ],
};

export const WithErrors: Story = {
  args: {
    pluginId: fixtureNerManifest.plugin_id,
    schema: fixtureNerManifest.config_schema,
    value: { model_id: "", entity_types: [] },
    errors: [{ path: ["model_id"], message: "Model selection is required" }],
  },
  decorators: [
    function PluginOptionsDecorator(Story) {
      const readyState: PluginOptionsState = {
        status: "ready",
        options: fixturePluginOptions["tabiya.ner.v1"].model_id.options,
        error: null,
      };
      return (
        <PluginOptionsOverrideContext.Provider value={readyState}>
          <Story />
        </PluginOptionsOverrideContext.Provider>
      );
    },
  ],
};
