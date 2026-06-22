/**
 * Centralized route paths. Importers reference these instead of string
 * literals so renames stay safe and feature code stays decoupled from the
 * URL shape.
 */

export const routerPaths = {
  ROOT: "/",
  LOGIN: "/login",
  DASHBOARD: "/dashboard",
  CONFIGURATION: "/configuration",
  KEYS: "/keys",
} as const;