import type { HttpHandler } from "msw";
import { apiKeysHandlers } from "./handlers/apiKeys";
import { classifyHandlers } from "./handlers/classify";
import { nelV2Handlers } from "./handlers/nelV2";
import { pipelinesHandlers } from "./handlers/pipelines";
import { pluginsHandlers } from "./handlers/plugins";

export const handlers: HttpHandler[] = [
  ...nelV2Handlers,
  ...apiKeysHandlers,
  ...classifyHandlers,
  ...pluginsHandlers,
  ...pipelinesHandlers,
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
  resetPipelinesHandlersStore,
  seedPipelinesHandlersStore,
} from "./handlers/pipelines";
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
export {
  fixtureDefaultTabiyaPipeline,
  fixtureDefaultTabiyaStages,
  fixturePipelines,
  fixtureRecruiterTuningPipeline,
  fixtureRecruiterTuningStages,
} from "./fixtures/pipelines";
export {
  fixtureNerManifest,
  fixtureNelManifest,
  fixturePluginDetails,
  fixturePluginManifests,
  fixturePluginOptions,
  fixturePluginSummaries,
  fixtureResultsManifest,
  fixtureTextInputManifest,
} from "./fixtures/plugins";
