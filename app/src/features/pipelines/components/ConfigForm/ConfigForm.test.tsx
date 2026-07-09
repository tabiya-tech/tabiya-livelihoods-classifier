import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  fixtureNerManifest,
  fixtureNelManifest,
  fixturePluginOptions,
} from "@/mocks/fixtures/plugins";
import type { PluginOptionsState } from "../../hooks/usePluginOptions";
import { PluginOptionsOverrideContext } from "../../hooks/pipelinesOverrides";
import { ConfigForm, DATA_TEST_ID } from "./ConfigForm";

// ── helpers ──────────────────────────────────────────────────────────────────

function renderWithPluginOptions(
  pluginOptionsState: PluginOptionsState,
  children: React.ReactNode,
) {
  return render(
    <PluginOptionsOverrideContext.Provider value={pluginOptionsState}>
      {children}
    </PluginOptionsOverrideContext.Provider>,
  );
}

// ── tests ────────────────────────────────────────────────────────────────────

describe("ConfigForm", () => {
  it("renders a text input for a plain string field", () => {
    // GIVEN a schema with one plain string field
    const givenSchema = {
      type: "object",
      properties: {
        title: { type: "string", title: "Title" },
      },
    };
    const givenValue = { title: "Hello world" };
    const givenOnChange = vi.fn();

    // WHEN the form is rendered
    render(
      <ConfigForm
        schema={givenSchema}
        value={givenValue}
        onChange={givenOnChange}
        pluginId="tabiya.test.v1"
      />,
    );

    // THEN an input of type text appears with the current value
    const renderedInput = screen.getByRole("textbox");
    expect(renderedInput).toBeInTheDocument();
    expect(renderedInput).toHaveAttribute("type", "text");
    expect(renderedInput).toHaveValue(givenValue.title);
  });

  it("renders a select with enum options for a string+enum field", () => {
    // GIVEN a schema with a string enum field
    const givenEnumValues = ["option-alpha", "option-beta", "option-gamma"];
    const givenSchema = {
      type: "object",
      properties: {
        mode: {
          type: "string",
          title: "Mode",
          enum: givenEnumValues,
        },
      },
    };
    const givenValue = { mode: "option-alpha" };
    const givenOnChange = vi.fn();

    // WHEN the form is rendered
    render(
      <ConfigForm
        schema={givenSchema}
        value={givenValue}
        onChange={givenOnChange}
        pluginId="tabiya.test.v1"
      />,
    );

    // THEN a select element appears with one option per enum value
    const renderedSelect = screen.getByRole("combobox");
    expect(renderedSelect).toBeInTheDocument();
    const renderedOptions = screen.getAllByRole("option");
    expect(renderedOptions).toHaveLength(givenEnumValues.length);
    givenEnumValues.forEach((enumValue) => {
      expect(screen.getByRole("option", { name: enumValue })).toBeInTheDocument();
    });
  });

  it("renders a range input for an integer field with min and max", () => {
    // GIVEN the NEL manifest which has top_k with minimum=1, maximum=50
    const givenSchema = fixtureNelManifest.config_schema;
    const givenValue = { top_k: 5, min_similarity: 0 };
    const givenOnChange = vi.fn();
    const givenExpectedMin = "1";
    const givenExpectedMax = "50";

    // WHEN the form is rendered (with a ready override context so x-source fields load)
    const givenReadyOptions: PluginOptionsState = {
      status: "ready",
      options: [],
      error: null,
    };
    renderWithPluginOptions(
      givenReadyOptions,
      <ConfigForm
        schema={givenSchema}
        value={givenValue}
        onChange={givenOnChange}
        pluginId={fixtureNelManifest.plugin_id}
      />,
    );

    // THEN an input[type=range] appears with the correct min and max
    const renderedRangeInputs = screen.getAllByRole("slider");
    expect(renderedRangeInputs.length).toBeGreaterThan(0);
    const topKSlider = renderedRangeInputs[0];
    expect(topKSlider).toHaveAttribute("min", givenExpectedMin);
    expect(topKSlider).toHaveAttribute("max", givenExpectedMax);
  });

  it("renders a toggle button with aria-checked for a boolean field", async () => {
    // GIVEN a schema with one boolean field, initially false
    const givenSchema = {
      type: "object",
      properties: {
        enabled: { type: "boolean", title: "Enabled" },
      },
    };
    const givenInitialValue = false;
    const givenOnChange = vi.fn();

    // WHEN the form is rendered
    render(
      <ConfigForm
        schema={givenSchema}
        value={{ enabled: givenInitialValue }}
        onChange={givenOnChange}
        pluginId="tabiya.test.v1"
      />,
    );

    // THEN a toggle (role=switch) is present with aria-checked matching the value
    const renderedToggle = screen.getByRole("switch");
    expect(renderedToggle).toBeInTheDocument();
    expect(renderedToggle).toHaveAttribute("aria-checked", "false");
  });

  it("renders chip buttons for an array+enum field and toggles values on click", async () => {
    // GIVEN the NER manifest which has entity_types array+enum
    const givenSchema = fixtureNerManifest.config_schema;
    const givenInitialSelected = ["occupation"];
    const givenOnChange = vi.fn();

    // AND a ready plugin options state so x-source fields do not block
    const givenReadyOptions: PluginOptionsState = {
      status: "ready",
      options: [],
      error: null,
    };

    // WHEN the form is rendered
    renderWithPluginOptions(
      givenReadyOptions,
      <ConfigForm
        schema={givenSchema}
        value={{ model_id: "", entity_types: givenInitialSelected }}
        onChange={givenOnChange}
        pluginId={fixtureNerManifest.plugin_id}
      />,
    );

    // THEN chip buttons appear for each enum option
    const renderedChips = screen.getAllByTestId(DATA_TEST_ID.ARRAY_CHIP);
    expect(renderedChips.length).toBeGreaterThan(0);

    // WHEN a chip that is not yet selected is clicked
    const givenUnselectedChipLabel = "skill";
    const unselectedChip = screen.getByRole("button", { name: givenUnselectedChipLabel });
    await userEvent.click(unselectedChip);

    // THEN onChange is called with the chip value added to the array
    expect(givenOnChange).toHaveBeenCalledWith(
      expect.objectContaining({
        entity_types: expect.arrayContaining([...givenInitialSelected, givenUnselectedChipLabel]),
      }),
    );
  });

  it("renders a select with dynamic options for an x-source field via PluginOptionsOverrideContext", async () => {
    // GIVEN the NER manifest (has model_id with x-source) and a ready options state
    const givenSchema = fixtureNerManifest.config_schema;
    const givenOptions = fixturePluginOptions["tabiya.ner.v1"].model_id.options;
    const givenReadyOptions: PluginOptionsState = {
      status: "ready",
      options: givenOptions,
      error: null,
    };
    const givenOnChange = vi.fn();
    const expectedFirstOptionLabel = givenOptions[0].label;

    // WHEN the form is rendered wrapped in the override context
    renderWithPluginOptions(
      givenReadyOptions,
      <ConfigForm
        schema={givenSchema}
        value={{ model_id: "", entity_types: [] }}
        onChange={givenOnChange}
        pluginId={fixtureNerManifest.plugin_id}
      />,
    );

    // THEN the select renders with the dynamic options from the context
    await waitFor(() => {
      const renderedOption = screen.getByRole("option", { name: expectedFirstOptionLabel });
      expect(renderedOption).toBeInTheDocument();
    });
  });

  it("shows the per-field error message when errors prop contains a matching path", () => {
    // GIVEN the NER schema and an error for model_id
    const givenSchema = fixtureNerManifest.config_schema;
    const givenErrors = [{ path: ["model_id"], message: "Required" }];
    const expectedErrorText = "Required";

    // AND a ready override so options load without MSW
    const givenReadyOptions: PluginOptionsState = {
      status: "ready",
      options: [],
      error: null,
    };

    // WHEN the form is rendered with the errors prop
    renderWithPluginOptions(
      givenReadyOptions,
      <ConfigForm
        schema={givenSchema}
        value={{ model_id: "", entity_types: [] }}
        onChange={vi.fn()}
        errors={givenErrors}
        pluginId={fixtureNerManifest.plugin_id}
      />,
    );

    // THEN the error message is visible in the document
    expect(screen.getByText(expectedErrorText)).toBeInTheDocument();
  });
});
