/**
 * MSW handlers for /v2/classifications and /v2/usage.
 */

import { http, HttpResponse } from "msw";
import { API_BASE_URL } from "@/lib/api";
import type { ClassificationSummary, DailyCount } from "@/lib/api";
import {
  fixtureClassificationSummaries,
  fixtureDailyCounts,
} from "../fixtures/classifications";

let currentClassifications: ClassificationSummary[] = [
  ...fixtureClassificationSummaries,
];
let currentDailyCounts: DailyCount[] = [...fixtureDailyCounts];

export function resetClassificationsHandlersStore() {
  currentClassifications = [...fixtureClassificationSummaries];
  currentDailyCounts = [...fixtureDailyCounts];
}

export function seedClassificationsHandlersStore(seed: {
  classifications?: ClassificationSummary[];
  dailyCounts?: DailyCount[];
}) {
  if (seed.classifications) currentClassifications = [...seed.classifications];
  if (seed.dailyCounts) currentDailyCounts = [...seed.dailyCounts];
}

const classificationsUrl = `${API_BASE_URL}/v2/classifications`;
const usageUrl = `${API_BASE_URL}/v2/usage`;

export const classificationsHandlers = [
  http.get(classificationsUrl, ({ request }) => {
    const url = new URL(request.url);
    const limit = Math.min(
      100,
      Math.max(1, Number(url.searchParams.get("limit") ?? 20)),
    );
    const cursor = url.searchParams.get("cursor") ?? null;

    let items = currentClassifications;
    if (cursor) {
      const cursorIndex = items.findIndex(
        (item) => item.classification_id === cursor,
      );
      items = cursorIndex >= 0 ? items.slice(cursorIndex + 1) : [];
    }

    const page = items.slice(0, limit);
    const nextCursor =
      items.length > limit ? page[page.length - 1].classification_id : null;

    return HttpResponse.json({ items: page, next_cursor: nextCursor });
  }),

  http.get(usageUrl, ({ request }) => {
    const url = new URL(request.url);
    const days = Number(url.searchParams.get("days") ?? 30);
    return HttpResponse.json({ days, data: currentDailyCounts });
  }),
];
