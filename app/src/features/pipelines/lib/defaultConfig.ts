/**
 * Builds the initial `config` for a freshly-dropped stage from its manifest's
 * JSON-Schema `default` values.
 *
 * Before this, dropped stages started with empty config `{}`, so any stage
 * with `required` fields (e.g. NEL's nel_model_id / taxonomy_model_id) failed
 * validation immediately — even in a correct text→ner→nel→sink chain — until
 * the user opened the drawer and picked values. Seeding declared defaults gets
 * optional fields (top_k, min_similarity, text_field, …) filled automatically;
 * required fields with no default are left absent so the required-field UI can
 * flag them.
 */

import type { PluginManifest } from "@/lib/api";

type JsonSchemaProperty = {
  default?: unknown;
};

export function defaultConfigForManifest(
  manifest: PluginManifest | undefined,
): Record<string, unknown> {
  const config: Record<string, unknown> = {};
  const schema = manifest?.config_schema as
    | { properties?: Record<string, JsonSchemaProperty> }
    | undefined;
  const properties = schema?.properties;
  if (!properties) {
    return config;
  }
  for (const [fieldName, fieldSchema] of Object.entries(properties)) {
    if (fieldSchema && "default" in fieldSchema) {
      config[fieldName] = fieldSchema.default;
    }
  }
  return config;
}

/**
 * Returns the names of `required` config fields that are missing or empty in
 * the given config — used to flag "needs configuration" on a stage.
 */
export function missingRequiredFields(
  manifest: PluginManifest | undefined,
  config: Record<string, unknown> | undefined,
): string[] {
  const schema = manifest?.config_schema as
    | { required?: string[] }
    | undefined;
  const required = schema?.required ?? [];
  const currentConfig = config ?? {};
  return required.filter((fieldName) => {
    const value = currentConfig[fieldName];
    return value === undefined || value === null || value === "";
  });
}
