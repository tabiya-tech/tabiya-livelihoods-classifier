/**
 * Centralized route paths. Importers reference these instead of string
 * literals so renames stay safe and feature code stays decoupled from the
 * URL shape.
 */

export const routerPaths = {
  ROOT: "/",
  LOGIN: "/login",
  DASHBOARD: "/dashboard",
  CLASSIFIER: "/classifier",
  PIPELINES: "/pipelines",
  PIPELINE_NEW: "/pipelines/new",
  PIPELINE_EDIT: "/pipelines/:pipelineId",
  CONFIGURATION: "/configuration",
  KEYS: "/keys",
} as const;