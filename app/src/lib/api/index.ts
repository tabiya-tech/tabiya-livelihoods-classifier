export { ApiError, API_BASE_URL, NEL_V2_API_BASE_URL, request } from "./fetcher";
export type { RequestContext, RequestOptions } from "./fetcher";

export {
  getHealth,
  getUserConfig,
  saveUserConfig,
  getUsage,
} from "./v1";
export type { HealthResponse, UsagePoint, UserConfig } from "./v1";

export {
  getV2UserConfig,
  saveV2UserConfig,
  listNelModels,
  listTaxonomyModels,
  listApiKeys,
  createApiKey,
  deleteApiKey,
  classify,
} from "./v2";
export type {
  NelModel,
  TaxonomyModel,
  V2UserConfig,
  ApiKeyMetadata,
  ListApiKeysResponse,
  CreateApiKeyResponse,
  ClassifyEntityType,
  ClassifyEntitySpan,
  ClassifyOccupationEntity,
  ClassifySkillEntity,
  ClassifyQualificationEntity,
  ClassifyMatchEntity,
  ClassifyOccupationMatch,
  ClassifySkillMatch,
  ClassifyQualificationMatch,
  ClassifyMatch,
  ClassifiedEntity,
  ClassifyOptions,
  ClassifyRequest,
  ClassifyMetadata,
  ClassifyResponse,
} from "./v2";
