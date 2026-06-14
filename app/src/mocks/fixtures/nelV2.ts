/**
 * Canonical fixtures for the NEL v2 backend. Used by both Storybook (via
 * msw-storybook-addon) and Vitest tests so stories and assertions stay in
 * lock-step.
 */

import type { NelModel, TaxonomyModel, V2UserConfig } from "@/lib/api";

export const fixtureNelModels: NelModel[] = [
  {
    model_id: "all-MiniLM-L6-v2",
    dimensions: 384,
    description:
      "Fast general-purpose sentence embedder. Good for high-throughput classification.",
  },
  {
    model_id: "mpnet-base-v2",
    dimensions: 768,
    description:
      "Higher quality embeddings; ~2× slower. Recommended for analyst workflows.",
  },
  {
    model_id: "tabiya-job-bge",
    dimensions: 1024,
    description:
      "Fine-tuned on job-ad corpora. Best precision on occupation linking.",
  },
];

export const fixtureTaxonomyModels: TaxonomyModel[] = [
  {
    id: "esco-1.1.1",
    name: "ESCO",
    version: "v1.1.1",
    description:
      "Default European Skills/Competences/Qualifications/Occupations.",
    released: true,
  },
  {
    id: "esco-1.2.0",
    name: "ESCO",
    version: "v1.2.0",
    description: "Latest ESCO release with 2024 occupation refresh.",
    released: true,
  },
  {
    id: "isco-08",
    name: "ISCO",
    version: "08",
    description: "ILO occupation classification, 4-digit codes.",
    released: true,
  },
];

export const fixtureUserConfig: V2UserConfig = {
  nel_model_id: fixtureNelModels[1].model_id,
  taxonomy_model_id: fixtureTaxonomyModels[1].id,
};
