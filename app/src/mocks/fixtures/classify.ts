/**
 * Canonical Classifier fixtures. Spans line up with the sample text so the
 * EntityHighlight render is exact — change the text and the spans must move
 * with it.
 */

import type { ClassifiedEntity, ClassifyResponse } from "@/lib/api";

export const fixtureClassifySourceText =
  "Senior data scientist with experience in Python, SQL, and machine learning. Bachelor's degree in Statistics required.";
//   0         1         2         3         4         5         6         7         8         9         0         1
//   0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567

export const fixtureClassifyEntities: ClassifiedEntity[] = [
  {
    entity_type: "occupation",
    surface_form: "data scientist",
    span: { start: 7, end: 21 },
    matches: [
      {
        entity_type: "occupation",
        similarity_score: 0.94,
        entity: {
          uuid: "occ-data-scientist",
          origin_uuid: "occ-data-scientist",
          uuid_history: [],
          preferred_label: "data scientist",
          origin_uri: "http://data.europa.eu/esco/occupation/data-scientist",
          alt_labels: ["data analyst", "research data scientist"],
          description:
            "Data scientists analyse complex datasets to extract actionable insights and build predictive models.",
          esco_code: "2511.4",
        },
      },
      {
        entity_type: "occupation",
        similarity_score: 0.81,
        entity: {
          uuid: "occ-data-analyst",
          origin_uuid: "occ-data-analyst",
          uuid_history: [],
          preferred_label: "data analyst",
          origin_uri: "http://data.europa.eu/esco/occupation/data-analyst",
          alt_labels: [],
          description:
            "Data analysts produce reports from large datasets and surface trends for stakeholders.",
          esco_code: "2511.3",
        },
      },
    ],
  },
  {
    entity_type: "skill",
    surface_form: "Python",
    span: { start: 41, end: 47 },
    matches: [
      {
        entity_type: "skill",
        similarity_score: 0.97,
        entity: {
          uuid: "skill-python",
          origin_uuid: "skill-python",
          uuid_history: [],
          preferred_label: "Python (computer programming)",
          origin_uri: "http://data.europa.eu/esco/skill/python",
          alt_labels: ["Python programming"],
          description:
            "The techniques and principles of software development using the Python programming language.",
          skill_type: "skill/competence",
          reuse_level: "cross-sector",
        },
      },
    ],
  },
  {
    entity_type: "skill",
    surface_form: "SQL",
    span: { start: 49, end: 52 },
    matches: [
      {
        entity_type: "skill",
        similarity_score: 0.93,
        entity: {
          uuid: "skill-sql",
          origin_uuid: "skill-sql",
          uuid_history: [],
          preferred_label: "SQL",
          origin_uri: "http://data.europa.eu/esco/skill/sql",
          alt_labels: ["Structured Query Language"],
          description:
            "Structured Query Language for relational database access and manipulation.",
          skill_type: "skill/competence",
          reuse_level: "cross-sector",
        },
      },
    ],
  },
  {
    entity_type: "skill",
    surface_form: "machine learning",
    span: { start: 58, end: 74 },
    matches: [
      {
        entity_type: "skill",
        similarity_score: 0.95,
        entity: {
          uuid: "skill-ml",
          origin_uuid: "skill-ml",
          uuid_history: [],
          preferred_label: "machine learning",
          origin_uri: "http://data.europa.eu/esco/skill/machine-learning",
          alt_labels: ["ML", "statistical learning"],
          description:
            "Algorithms and statistical models that allow computer systems to improve from data without explicit programming.",
          skill_type: "skill/competence",
          reuse_level: "sector-specific",
        },
      },
      {
        entity_type: "skill",
        similarity_score: 0.78,
        entity: {
          uuid: "skill-ai",
          origin_uuid: "skill-ai",
          uuid_history: [],
          preferred_label: "artificial intelligence",
          origin_uri: "http://data.europa.eu/esco/skill/ai",
          alt_labels: ["AI"],
          description: "Theory and methods of building intelligent agents.",
          skill_type: "skill/competence",
          reuse_level: "cross-sector",
        },
      },
    ],
  },
  {
    entity_type: "qualification",
    surface_form: "Bachelor's degree in Statistics",
    span: { start: 76, end: 107 },
    matches: [
      {
        entity_type: "qualification",
        similarity_score: 0.9,
        entity: {
          uuid: "qual-bsc-stats",
          origin_uuid: "qual-bsc-stats",
          uuid_history: [],
          preferred_label: "Bachelor of Statistics",
          origin_uri: "http://data.europa.eu/esco/qualification/bsc-stats",
          alt_labels: ["BSc Statistics", "BS Statistics"],
          description:
            "Undergraduate degree covering statistical theory and applied data analysis.",
          eqf_level: "6",
          country: "EU",
        },
      },
    ],
  },
];

export const fixtureClassifyResponse: ClassifyResponse = {
  entities: fixtureClassifyEntities,
  metadata: {
    classifier_version: "2.0.0",
    ner_model: "tabiya/roberta-base-job-ner",
    nel_model_id: "all-MiniLM-L6-v2",
    taxonomy_model_id: "68934c97fb6143f42db01da5",
    processing_time_ms: 412,
  },
};
