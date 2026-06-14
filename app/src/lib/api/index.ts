export { ApiError, API_BASE_URL, NEL_V2_API_BASE_URL, request } from "./fetcher";
export type { RequestContext, RequestOptions } from "./fetcher";

export {
  getHealth,
  getUserConfig,
  saveUserConfig,
  listApiKeys,
  createApiKey,
  deleteApiKey,
  getUsage,
} from "./v1";
export type { ApiKey, HealthResponse, UsagePoint, UserConfig } from "./v1";

export {
  getV2UserConfig,
  saveV2UserConfig,
  listNelModels,
  listTaxonomyModels,
} from "./v2";
export type { NelModel, TaxonomyModel, V2UserConfig } from "./v2";
