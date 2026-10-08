/**
 * MSW handlers for /v2/plugins. Read-only — no store, no reset helper.
 */

import { http, HttpResponse } from "msw";
import { API_BASE_URL } from "@/lib/api";
import {
  fixturePluginDetails,
  fixturePluginOptions,
  fixturePluginSummaries,
} from "../fixtures/plugins";

const listUrl = `${API_BASE_URL}/v2/plugins`;
const detailUrl = `${API_BASE_URL}/v2/plugins/:pluginId`;
const optionsUrl = `${API_BASE_URL}/v2/plugins/:pluginId/options/:field`;

export const pluginsHandlers = [
  http.get(listUrl, () =>
    HttpResponse.json({ plugins: fixturePluginSummaries }),
  ),
  http.get(detailUrl, ({ params }) => {
    const pluginId = String(params.pluginId);
    const detail = fixturePluginDetails[pluginId];
    if (!detail) {
      return HttpResponse.json({ detail: "Plugin not found" }, { status: 404 });
    }
    return HttpResponse.json(detail);
  }),
  http.get(optionsUrl, ({ params }) => {
    const pluginId = String(params.pluginId);
    const field = String(params.field);
    const options = fixturePluginOptions[pluginId]?.[field];
    if (!options) {
      return HttpResponse.json(
        { detail: `No options for ${pluginId}.${field}` },
        { status: 404 },
      );
    }
    return HttpResponse.json(options);
  }),
];
