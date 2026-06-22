import type { HttpHandler } from "msw";
import { apiKeysHandlers } from "./handlers/apiKeys";
import { classifyHandlers } from "./handlers/classify";
import { nelV2Handlers } from "./handlers/nelV2";

export const handlers: HttpHandler[] = [
  ...nelV2Handlers,
  ...apiKeysHandlers,
  ...classifyHandlers,
];

export { resetNelV2HandlersStore } from "./handlers/nelV2";
export {
  resetApiKeysHandlersStore,
  seedApiKeysHandlersStore,
} from "./handlers/apiKeys";
export {
  resetClassifyHandlersStore,
  seedClassifyHandlersStore,
} from "./handlers/classify";
export {
  fixtureNelModels,
  fixtureTaxonomyModels,
  fixtureUserConfig,
} from "./fixtures/nelV2";
export { fixtureApiKeyMetadata, fixtureCreatedApiKey } from "./fixtures/apiKeys";
export {
  fixtureClassifyEntities,
  fixtureClassifyResponse,
  fixtureClassifySourceText,
} from "./fixtures/classify";
