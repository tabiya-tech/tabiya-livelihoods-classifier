/**
 * Storybook decorators that wrap a story in the providers / router context
 * production components depend on, with deterministic mock values.
 *
 * Why this lives in _test_utilities (not in a story file): multiple page
 * stories need the same setup, and Storybook decorators are framework code,
 * not test code — but the data they inject is the same flavor of mocking we
 * use elsewhere, so co-locating with the test helpers keeps the
 * "fake-data plumbing" in one place.
 */

import type { ReactElement } from "react";
import { MemoryRouter } from "react-router-dom";
import {
  AuthOverrideProvider,
  type UseFirebaseAuthValue,
  type AuthenticatedUser,
} from "@/lib/auth/useFirebaseAuth";

type StoryRenderer = () => ReactElement;

export interface RouterStoryOptions {
  /** Initial pathname for MemoryRouter. Defaults to "/". */
  initialPath?: string;
}

/** Wrap a story in a MemoryRouter at the given path. */
export function withRouter({ initialPath = "/" }: RouterStoryOptions = {}) {
  function StoryWithRouter(Story: StoryRenderer) {
    return (
      <MemoryRouter initialEntries={[initialPath]}>
        <Story />
      </MemoryRouter>
    );
  }
  return StoryWithRouter;
}

export interface FirebaseAuthStoryOptions {
  /** The signed-in user, or null for signed-out. Defaults to null. */
  user?: AuthenticatedUser | null;
  /** Whether the hook should report initial auth-resolution loading. Defaults to false. */
  loading?: boolean;
}

/**
 * Wrap a story in an AuthOverrideProvider so useFirebaseAuth returns a
 * deterministic value. Sign-in/up/out callbacks are no-ops by default.
 */
export function withFirebaseAuth({
  user = null,
  loading = false,
}: FirebaseAuthStoryOptions = {}) {
  const value: UseFirebaseAuthValue = {
    user,
    loading,
    signInWithEmail: async () => undefined,
    signUpWithEmail: async () => undefined,
    signOut: async () => undefined,
  };
  function StoryWithFirebaseAuth(Story: StoryRenderer) {
    return (
      <AuthOverrideProvider value={value}>
        <Story />
      </AuthOverrideProvider>
    );
  }
  return StoryWithFirebaseAuth;
}
