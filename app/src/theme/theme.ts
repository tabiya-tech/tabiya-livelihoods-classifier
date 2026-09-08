/**
 * Design tokens — single source of truth for the Tabiya Classifier UI.
 *
 * Imported by tailwind.config.ts (for utility class generation) and by
 * non-Tailwind consumers like Recharts that need the raw values.
 *
 * Mirrors the CSS variables defined in the design handoff styles.css.
 */

export const colors = {
  navy: { DEFAULT: "#002147", 700: "#0a2a55" },
  ink: "#0c1a2e",
  lime: { DEFAULT: "#00ff91", 600: "#00d579" },
  yellow: "#eeff41",
  teal: "#26887d",
  cream: { DEFAULT: "#f3f1ee", 200: "#ece9e3" },
  paper: "#faf9f6",
  line: { DEFAULT: "#e0ddd9", strong: "#c9c5be" },
  muted: { DEFAULT: "#6b6b6b", 2: "#8a8780" },
  error: "#c0392b",
  entity: {
    occupation: { fg: "#002147", bg: "#d9e3f0" },
    skill: { fg: "#26887d", bg: "#d4ebe6" },
    qualification: { fg: "#b8860b", bg: "#f3ecc7" },
    experience: { fg: "#7a3e9d", bg: "#ebd9f4" },
    domain: { fg: "#b03a4a", bg: "#f4d8dc" },
  },
} as const;

export const fontFamily: Record<"mono" | "sans" | "serif", string[]> = {
  mono: ['"IBM Plex Mono"', "ui-monospace", '"SF Mono"', "Menlo", "monospace"],
  sans: ["Inter", "system-ui", "-apple-system", '"Segoe UI"', "sans-serif"],
  serif: ['"Source Serif 4"', '"Source Serif Pro"', "Georgia", "serif"],
};

export const borderRadius = {
  sm: "4px",
  DEFAULT: "6px",
  md: "10px",
  lg: "14px",
} as const;

export const boxShadow = {
  "card-1":
    "0 1px 0 rgba(12, 26, 46, 0.04), 0 1px 2px rgba(12, 26, 46, 0.04)",
  "card-2":
    "0 1px 0 rgba(12, 26, 46, 0.04), 0 4px 16px rgba(12, 26, 46, 0.06)",
} as const;

/** Entity types — keep in sync with colors.entity keys. */
export const ENTITY_TYPES = [
  "occupation",
  "skill",
  "qualification",
  "experience",
  "domain",
] as const;

export type EntityType = (typeof ENTITY_TYPES)[number];
