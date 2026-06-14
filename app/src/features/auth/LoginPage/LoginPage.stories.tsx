import type { Decorator, Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { LoginPage } from "./LoginPage";
import {
  AuthOverrideProvider,
  type UseFirebaseAuthValue,
} from "@/lib/auth/useFirebaseAuth";
import { withRouter } from "@/_test_utilities";

const meta: Meta<typeof LoginPage> = {
  title: "Pages/LoginPage",
  component: LoginPage,
  parameters: { layout: "fullscreen" },
};
export default meta;

type Story = StoryObj<typeof LoginPage>;

function buildAuthOverride(
  overrides: Partial<UseFirebaseAuthValue> = {},
): UseFirebaseAuthValue {
  return {
    user: null,
    loading: false,
    signInWithEmail: fn(async () => undefined),
    signUpWithEmail: fn(async () => undefined),
    signOut: fn(async () => undefined),
    ...overrides,
  };
}

function withAuthOverride(value: UseFirebaseAuthValue): Decorator {
  function StoryWithAuthOverride(Story: () => React.ReactElement) {
    return (
      <AuthOverrideProvider value={value}>
        <Story />
      </AuthOverrideProvider>
    );
  }
  return StoryWithAuthOverride;
}

const loginRouterDecorator = withRouter({ initialPath: "/login" });

export const Default: Story = {
  decorators: [loginRouterDecorator, withAuthOverride(buildAuthOverride())],
};

export const SignInRejected: Story = {
  decorators: [
    loginRouterDecorator,
    withAuthOverride(
      buildAuthOverride({
        signInWithEmail: fn(async () => {
          throw new Error("auth/invalid-credential");
        }),
      }),
    ),
  ],
};

export const SignInPending: Story = {
  decorators: [
    loginRouterDecorator,
    withAuthOverride(
      buildAuthOverride({
        // Resolves only after the story is no longer interesting — keeps the
        // submit button in its loading state for the duration of the story.
        signInWithEmail: fn(() => new Promise<void>(() => {})),
      }),
    ),
  ],
};
