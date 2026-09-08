import type { Meta, StoryObj } from "@storybook/react";
import { DashboardPage } from "./DashboardPage";
import { withFirebaseAuth } from "@/_test_utilities";

const meta: Meta<typeof DashboardPage> = {
  title: "Pages/DashboardPage",
  component: DashboardPage,
  parameters: { layout: "fullscreen" },
};
export default meta;

type Story = StoryObj<typeof DashboardPage>;

export const SignedIn: Story = {
  decorators: [
    withFirebaseAuth({
      user: {
        id: "uid-sara",
        email: "sara.m@tabiya.org",
        initials: "SM",
      },
    }),
  ],
};

/**
 * Defensive fallback: the page should still render legibly even when the auth
 * hook has no user (e.g. mid sign-out transition). Useful for catching
 * regressions where displayName goes blank or crashes the page.
 */
export const NoUserFallback: Story = {
  decorators: [withFirebaseAuth({ user: null })],
};
