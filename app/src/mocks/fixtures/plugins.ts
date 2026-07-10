/**
 * Canonical fixtures for /v2/plugins. Six-plugin catalog matching what
 * the backend ships in `backend/classify_v2/.../pipelines/registry/catalog.json`.
 *
 * Two of the six are `coming_soon` placeholders — palette renders them as
 * greyed-out cards; validator rejects any pipeline referencing them.
 */

import type {
  PluginDetail,
  PluginManifest,
  PluginOptionsResponse,
  PluginSummary,
} from "@/lib/api";

const CONTRACT_VERSION = "1.0.0";

export const fixtureTextInputManifest: PluginManifest = {
  plugin_id: "tabiya.source.text.v1",
  name: "Text Input",
  version: "0.1.0",
  category: "source",
  summary: "Feeds raw text into the pipeline.",
  detail: "paste, upload, or title + description",
  icon: "text",
  input_slot: { type: "None", cardinality: "none" },
  output_slot: { type: "RawText", cardinality: "single" },
  config_schema: {
    type: "object",
    properties: {
      text: {
        type: "string",
        title: "Text",
        description:
          "Body text. Provide this OR title/description, not both.",
      },
      title: { type: "string", title: "Title" },
      description: { type: "string", title: "Description" },
    },
    additionalProperties: false,
  },
  timeout_ms: 5_000,
  "x-tabiya-contract-version": CONTRACT_VERSION,
};

export const fixtureNerManifest: PluginManifest = {
  plugin_id: "tabiya.ner.v1",
  name: "Tabiya NER",
  version: "0.1.0",
  category: "core",
  summary: "Named-entity recognition over job-ad prose.",
  detail: "roberta-base-job-ner",
  icon: "ner",
  input_slot: { type: "RawText", cardinality: "single" },
  output_slot: { type: "Entities", cardinality: "single" },
  config_schema: {
    type: "object",
    properties: {
      model_id: {
        type: "string",
        title: "Model",
        default: "tabiya/roberta-base-job-ner",
        "x-source": "/v2/plugins/tabiya.ner.v1/options/model_id",
      },
      entity_types: {
        type: "array",
        title: "Entity types",
        items: {
          type: "string",
          enum: ["occupation", "skill", "qualification", "experience", "domain"],
        },
      },
    },
    additionalProperties: false,
  },
  timeout_ms: 30_000,
  "x-tabiya-contract-version": CONTRACT_VERSION,
};

export const fixtureNelManifest: PluginManifest = {
  plugin_id: "tabiya.nel.v1",
  name: "Tabiya NEL",
  version: "0.1.0",
  category: "core",
  summary: "Links extracted entities to the ESCO taxonomy.",
  detail: "MongoDB Atlas vector search",
  icon: "nel",
  input_slot: { type: "Entities", cardinality: "single" },
  output_slot: { type: "LinkedEntities", cardinality: "single" },
  config_schema: {
    type: "object",
    properties: {
      nel_model_id: {
        type: "string",
        title: "Embedding model",
        "x-source": "/v2/plugins/tabiya.nel.v1/options/nel_model_id",
      },
      taxonomy_model_id: {
        type: "string",
        title: "Taxonomy model",
        "x-source": "/v2/plugins/tabiya.nel.v1/options/taxonomy_model_id",
      },
      top_k: {
        type: "integer",
        title: "Top K",
        minimum: 1,
        maximum: 50,
        default: 5,
      },
      min_similarity: {
        type: "number",
        title: "Min similarity",
        minimum: 0,
        maximum: 1,
        default: 0,
      },
    },
    required: ["nel_model_id", "taxonomy_model_id"],
    additionalProperties: false,
  },
  timeout_ms: 45_000,
  "x-tabiya-contract-version": CONTRACT_VERSION,
};

export const fixtureResultsManifest: PluginManifest = {
  plugin_id: "tabiya.sink.results.v1",
  name: "Results",
  version: "0.1.0",
  category: "sink",
  summary: "Consumes linked entities for downstream display.",
  icon: "results",
  input_slot: { type: "LinkedEntities", cardinality: "single" },
  output_slot: { type: "None", cardinality: "none" },
  config_schema: {
    type: "object",
    properties: {},
    additionalProperties: false,
  },
  timeout_ms: 5_000,
  "x-tabiya-contract-version": CONTRACT_VERSION,
};

/** Manifests keyed by plugin_id for fast lookup by fixture consumers. */
export const fixturePluginManifests: Record<string, PluginManifest> = {
  [fixtureTextInputManifest.plugin_id]: fixtureTextInputManifest,
  [fixtureNerManifest.plugin_id]: fixtureNerManifest,
  [fixtureNelManifest.plugin_id]: fixtureNelManifest,
  [fixtureResultsManifest.plugin_id]: fixtureResultsManifest,
};

function summaryFromManifest(manifest: PluginManifest): PluginSummary {
  return {
    plugin_id: manifest.plugin_id,
    name: manifest.name,
    version: manifest.version,
    category: manifest.category,
    summary: manifest.summary,
    detail: manifest.detail ?? null,
    icon: manifest.icon,
    status: "enabled",
    coming_soon: false,
    last_error: null,
  };
}

// Coming-soon plugins now ship real backend manifests, so their summaries
// carry a proper category (they land in the right palette section, not
// "Other"). They stay status=unavailable + coming_soon so the palette greys
// them and they can't be dropped.
const scraperSummary: PluginSummary = {
  plugin_id: "tabiya.source.scraper.v1",
  name: "Job Scraper",
  version: "0.1.0",
  category: "source",
  summary: "Fetches a job posting from a URL and feeds its text into the pipeline.",
  detail: "coming soon",
  icon: "globe",
  status: "unavailable",
  coming_soon: true,
  last_error: "coming_soon",
};

const stopwordsSummary: PluginSummary = {
  plugin_id: "tabiya.transform.stopwords.v1",
  name: "Stop-word Filter",
  version: "0.1.0",
  category: "transform",
  summary: "Removes common stop words from the text to cut down on noise entities.",
  detail: "coming soon",
  icon: "filter",
  status: "unavailable",
  coming_soon: true,
  last_error: "coming_soon",
};

const databaseSummary: PluginSummary = {
  plugin_id: "tabiya.sink.database.v1",
  name: "Database Sink",
  version: "0.1.0",
  category: "sink",
  summary: "Writes the linked entities to a configured database.",
  detail: "coming soon",
  icon: "download",
  status: "unavailable",
  coming_soon: true,
  last_error: "coming_soon",
};

const languageRouterSummary: PluginSummary = {
  plugin_id: "tabiya.branching.language_router.v1",
  name: "Language Router",
  version: "0.1.0",
  category: "core",
  summary: "Detects the input language and routes to a language-specific branch.",
  detail: "coming soon",
  icon: "globe",
  status: "unavailable",
  coming_soon: true,
  last_error: "coming_soon",
};

/** The full palette. Order matches the backend catalog. */
export const fixturePluginSummaries: PluginSummary[] = [
  summaryFromManifest(fixtureNerManifest),
  summaryFromManifest(fixtureNelManifest),
  summaryFromManifest(fixtureTextInputManifest),
  summaryFromManifest(fixtureResultsManifest),
  scraperSummary,
  stopwordsSummary,
  databaseSummary,
  languageRouterSummary,
];

export const fixturePluginDetails: Record<string, PluginDetail> = {
  ...Object.fromEntries(
    Object.values(fixturePluginManifests).map((manifest) => [
      manifest.plugin_id,
      {
        plugin_id: manifest.plugin_id,
        status: "enabled",
        coming_soon: false,
        last_error: null,
        manifest,
      } satisfies PluginDetail,
    ]),
  ),
  [scraperSummary.plugin_id]: {
    plugin_id: scraperSummary.plugin_id,
    status: "unavailable",
    coming_soon: true,
    last_error: "coming_soon",
    manifest: null,
  },
  [stopwordsSummary.plugin_id]: {
    plugin_id: stopwordsSummary.plugin_id,
    status: "unavailable",
    coming_soon: true,
    last_error: "coming_soon",
    manifest: null,
  },
  [databaseSummary.plugin_id]: {
    plugin_id: databaseSummary.plugin_id,
    status: "unavailable",
    coming_soon: true,
    last_error: "coming_soon",
    manifest: null,
  },
  [languageRouterSummary.plugin_id]: {
    plugin_id: languageRouterSummary.plugin_id,
    status: "unavailable",
    coming_soon: true,
    last_error: "coming_soon",
    manifest: null,
  },
};

/** Deterministic options each dropdown returns. Shared between test + story. */
export const fixturePluginOptions: Record<
  string,
  Record<string, PluginOptionsResponse>
> = {
  "tabiya.ner.v1": {
    model_id: {
      field: "model_id",
      options: [
        { value: "tabiya/roberta-base-job-ner", label: "Roberta base (job NER)" },
        { value: "tabiya/roberta-large-job-ner", label: "Roberta large (job NER)" },
      ],
    },
  },
  "tabiya.nel.v1": {
    nel_model_id: {
      field: "nel_model_id",
      options: [
        { value: "all-MiniLM-L6-v2", label: "all-MiniLM-L6-v2" },
        { value: "sentence-t5-base", label: "sentence-t5-base" },
      ],
    },
    taxonomy_model_id: {
      field: "taxonomy_model_id",
      options: [
        { value: "esco-v1.1", label: "ESCO v1.1" },
        { value: "esco-v1.2", label: "ESCO v1.2" },
      ],
    },
  },
};
