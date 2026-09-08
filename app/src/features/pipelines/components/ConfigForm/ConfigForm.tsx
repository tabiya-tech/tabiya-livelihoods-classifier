/**
 * Renders a JSON-Schema-driven configuration form for a pipeline stage.
 *
 * Each property in the schema's `properties` object becomes one form field.
 * The field type is determined by the JSON Schema type + optional keywords
 * (`enum`, `x-source`, `minimum`/`maximum`).
 */

import { useTranslation } from "react-i18next";
import { FormField, Input, Select, Slider, Toggle } from "@/components";
import { usePluginOptions } from "../../hooks/usePluginOptions";

const uniqueId = "a1b2c3d4-e5f6-4789-abcd-ef0123456789";

export const DATA_TEST_ID = {
  ROOT: `config-form-root-${uniqueId}`,
  FIELD: `config-form-field-${uniqueId}`,
  ARRAY_CHIP: `config-form-array-chip-${uniqueId}`,
  X_SOURCE_ERROR: `config-form-x-source-error-${uniqueId}`,
};

export interface ConfigFormFieldError {
  path: string[];
  message: string;
}

export interface ConfigFormProps {
  /** Plugin manifest's config_schema (JSON Schema object). */
  schema: Record<string, unknown>;
  /** Current config values. */
  value: Record<string, unknown>;
  onChange: (nextValue: Record<string, unknown>) => void;
  /** Per-field validation errors. */
  errors?: ConfigFormFieldError[];
  /** Used to fetch dynamic options for x-source fields. */
  pluginId: string;
}

// ── Internal per-field renderer ──────────────────────────────────────────────

interface FieldSchema {
  type?: string;
  title?: string;
  description?: string;
  enum?: unknown[];
  "x-source"?: string;
  minimum?: number;
  maximum?: number;
  items?: {
    type?: string;
    enum?: unknown[];
  };
}

interface FieldRendererProps {
  fieldName: string;
  fieldSchema: FieldSchema;
  fieldValue: unknown;
  fieldError: string | undefined;
  pluginId: string;
  onChange: (nextValue: Record<string, unknown>) => void;
  allValues: Record<string, unknown>;
}

function StringEnumField({
  fieldName,
  fieldSchema,
  fieldValue,
  onChange,
  allValues,
}: Omit<FieldRendererProps, "pluginId" | "fieldError">) {
  const enumValues = (fieldSchema.enum ?? []) as string[];
  return (
    <Select
      value={String(fieldValue ?? "")}
      onChange={(event) =>
        onChange({ ...allValues, [fieldName]: event.target.value })
      }
    >
      {enumValues.map((enumOption) => (
        <option key={enumOption} value={enumOption}>
          {enumOption}
        </option>
      ))}
    </Select>
  );
}

function XSourceField({
  fieldName,
  fieldValue,
  pluginId,
  onChange,
  allValues,
}: Omit<FieldRendererProps, "fieldSchema" | "fieldError">) {
  const { t } = useTranslation();
  const { status, options, error } = usePluginOptions(pluginId, fieldName);

  if (status === "error") {
    return (
      <p
        data-testid={DATA_TEST_ID.X_SOURCE_ERROR}
        style={{ color: "#b91c1c", fontSize: "12px" }}
      >
        {error?.message ?? t("pipelines.editor.configForm.errorOptions")}
      </p>
    );
  }

  if (status === "loading" || status === "idle") {
    return (
      <Select disabled value="">
        <option value="">{t("pipelines.editor.configForm.loading")}</option>
      </Select>
    );
  }

  return (
    <Select
      value={String(fieldValue ?? "")}
      onChange={(event) =>
        onChange({ ...allValues, [fieldName]: event.target.value })
      }
    >
      <option value="" disabled>
        {t("pipelines.editor.configForm.selectPlaceholder")}
      </option>
      {options.map((optionItem) => (
        <option key={optionItem.value} value={optionItem.value}>
          {optionItem.label}
        </option>
      ))}
    </Select>
  );
}

function NumberSliderField({
  fieldName,
  fieldSchema,
  fieldValue,
  onChange,
  allValues,
}: Omit<FieldRendererProps, "pluginId" | "fieldError">) {
  const minimum = fieldSchema.minimum ?? 0;
  const maximum = fieldSchema.maximum ?? 100;
  const step = fieldSchema.type === "integer" ? 1 : 0.01;

  return (
    <Slider
      min={minimum}
      max={maximum}
      step={step}
      value={Number(fieldValue ?? minimum)}
      onChange={(event) =>
        onChange({
          ...allValues,
          [fieldName]:
            fieldSchema.type === "integer"
              ? parseInt(event.target.value, 10)
              : parseFloat(event.target.value),
        })
      }
    />
  );
}

function ArrayEnumChipsField({
  fieldName,
  fieldSchema,
  fieldValue,
  onChange,
  allValues,
}: Omit<FieldRendererProps, "pluginId" | "fieldError">) {
  const enumValues = (fieldSchema.items?.enum ?? []) as string[];
  const selectedValues = Array.isArray(fieldValue) ? (fieldValue as string[]) : [];

  function handleChipToggle(chipValue: string) {
    const isSelected = selectedValues.includes(chipValue);
    const nextSelected = isSelected
      ? selectedValues.filter((existingValue) => existingValue !== chipValue)
      : [...selectedValues, chipValue];
    onChange({ ...allValues, [fieldName]: nextSelected });
  }

  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
      {enumValues.map((chipValue) => {
        const isSelected = selectedValues.includes(chipValue);
        return (
          <button
            key={chipValue}
            type="button"
            data-testid={DATA_TEST_ID.ARRAY_CHIP}
            onClick={() => handleChipToggle(chipValue)}
            style={{
              backgroundColor: isSelected ? "#0c1a2e" : "#e0ddd9",
              color: isSelected ? "#faf9f6" : "#0c1a2e",
              borderRadius: "10px",
              padding: "2px 10px",
              fontSize: "12px",
              border: "none",
              cursor: "pointer",
            }}
          >
            {chipValue}
          </button>
        );
      })}
    </div>
  );
}

function renderField(props: FieldRendererProps): React.ReactNode {
  const { fieldName, fieldSchema, fieldValue, pluginId, onChange, allValues } = props;
  const fieldType = fieldSchema.type;

  if (fieldType === "string") {
    if (fieldSchema.enum && Array.isArray(fieldSchema.enum)) {
      return (
        <StringEnumField
          fieldName={fieldName}
          fieldSchema={fieldSchema}
          fieldValue={fieldValue}
          onChange={onChange}
          allValues={allValues}
        />
      );
    }

    if (fieldSchema["x-source"]) {
      return (
        <XSourceField
          fieldName={fieldName}
          fieldValue={fieldValue}
          pluginId={pluginId}
          onChange={onChange}
          allValues={allValues}
        />
      );
    }

    return (
      <Input
        type="text"
        value={String(fieldValue ?? "")}
        onChange={(event) =>
          onChange({ ...allValues, [fieldName]: event.target.value })
        }
      />
    );
  }

  if (fieldType === "integer" || fieldType === "number") {
    const hasMinimum = fieldSchema.minimum !== undefined;
    const hasMaximum = fieldSchema.maximum !== undefined;

    if (hasMinimum && hasMaximum) {
      return (
        <NumberSliderField
          fieldName={fieldName}
          fieldSchema={fieldSchema}
          fieldValue={fieldValue}
          onChange={onChange}
          allValues={allValues}
        />
      );
    }

    return (
      <Input
        type="number"
        value={String(fieldValue ?? "")}
        onChange={(event) =>
          onChange({ ...allValues, [fieldName]: event.target.valueAsNumber })
        }
      />
    );
  }

  if (fieldType === "boolean") {
    return (
      <Toggle
        checked={Boolean(fieldValue)}
        onChange={(isChecked) =>
          onChange({ ...allValues, [fieldName]: isChecked })
        }
      />
    );
  }

  if (fieldType === "array") {
    const itemsSchema = fieldSchema.items;
    if (itemsSchema?.enum && Array.isArray(itemsSchema.enum)) {
      return (
        <ArrayEnumChipsField
          fieldName={fieldName}
          fieldSchema={fieldSchema}
          fieldValue={fieldValue}
          onChange={onChange}
          allValues={allValues}
        />
      );
    }
  }

  return null;
}

// ── Public component ──────────────────────────────────────────────────────────

export function ConfigForm({
  schema,
  value,
  onChange,
  errors,
  pluginId,
}: ConfigFormProps) {
  const { t } = useTranslation();
  const schemaProperties = (
    (schema.properties as Record<string, unknown>) ?? {}
  ) as Record<string, FieldSchema>;

  const propertyEntries = Object.entries(schemaProperties);

  if (propertyEntries.length === 0) {
    return (
      <div data-testid={DATA_TEST_ID.ROOT}>
        <p style={{ fontSize: "13px", color: "#6b6b6b" }}>
          {t("pipelines.editor.configForm.noOptions")}
        </p>
      </div>
    );
  }

  return (
    <div data-testid={DATA_TEST_ID.ROOT} style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      {propertyEntries.map(([fieldName, fieldSchema]) => {
        const fieldValue = value[fieldName];
        const fieldError = errors?.find(
          (errorItem) => errorItem.path[0] === fieldName,
        )?.message;

        const renderedControl = renderField({
          fieldName,
          fieldSchema,
          fieldValue,
          fieldError,
          pluginId,
          onChange,
          allValues: value,
        });

        if (renderedControl === null) {
          return null;
        }

        return (
          <FormField
            key={fieldName}
            data-testid={DATA_TEST_ID.FIELD}
            label={fieldSchema.title ?? fieldName}
            help={fieldSchema.description}
            error={fieldError}
          >
            {renderedControl}
          </FormField>
        );
      })}
    </div>
  );
}
