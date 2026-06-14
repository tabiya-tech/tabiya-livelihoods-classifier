import type { HttpHandler } from "msw";
import { nelV2Handlers } from "./handlers/nelV2";

export const handlers: HttpHandler[] = [...nelV2Handlers];

export { resetNelV2HandlersStore } from "./handlers/nelV2";
export {
  fixtureNelModels,
  fixtureTaxonomyModels,
  fixtureUserConfig,
} from "./fixtures/nelV2";
