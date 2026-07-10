/**
 * Canonical fixtures for /v2/classifications and /v2/usage.
 */

import type { ClassificationSummary, DailyCount } from "@/lib/api";

export const fixtureClassificationSummaries: ClassificationSummary[] = [
  {
    classification_id: "cls-001",
    pipeline_id: "pipeline-default",
    entity_count: 18,
    processing_time_ms: 342.5,
    created_at: "2026-07-10T10:00:00.000Z",
  },
  {
    classification_id: "cls-002",
    pipeline_id: "pipeline-default",
    entity_count: 11,
    processing_time_ms: 289.1,
    created_at: "2026-07-09T14:32:00.000Z",
  },
  {
    classification_id: "cls-003",
    pipeline_id: "pipeline-recruiter",
    entity_count: 7,
    processing_time_ms: 195.8,
    created_at: "2026-07-09T09:15:00.000Z",
  },
  {
    classification_id: "cls-004",
    pipeline_id: "pipeline-default",
    entity_count: 22,
    processing_time_ms: 401.2,
    created_at: "2026-07-08T16:44:00.000Z",
  },
  {
    classification_id: "cls-005",
    pipeline_id: "pipeline-default",
    entity_count: 9,
    processing_time_ms: 210.0,
    created_at: "2026-07-07T11:20:00.000Z",
  },
];

export const fixtureDailyCounts: DailyCount[] = [
  { date: "2026-06-10", count: 2 },
  { date: "2026-06-11", count: 5 },
  { date: "2026-06-12", count: 3 },
  { date: "2026-06-14", count: 8 },
  { date: "2026-06-15", count: 6 },
  { date: "2026-06-17", count: 4 },
  { date: "2026-06-18", count: 11 },
  { date: "2026-06-19", count: 7 },
  { date: "2026-06-21", count: 3 },
  { date: "2026-06-22", count: 9 },
  { date: "2026-06-24", count: 5 },
  { date: "2026-06-25", count: 14 },
  { date: "2026-06-26", count: 8 },
  { date: "2026-06-28", count: 6 },
  { date: "2026-06-29", count: 10 },
  { date: "2026-07-01", count: 4 },
  { date: "2026-07-02", count: 7 },
  { date: "2026-07-03", count: 12 },
  { date: "2026-07-05", count: 9 },
  { date: "2026-07-06", count: 5 },
  { date: "2026-07-07", count: 8 },
  { date: "2026-07-08", count: 15 },
  { date: "2026-07-09", count: 11 },
  { date: "2026-07-10", count: 3 },
];
