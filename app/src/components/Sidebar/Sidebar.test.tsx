import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Sidebar, DATA_TEST_ID } from "./Sidebar";
import { NAV_LINK_DATA_TEST_ID } from "@/components";

const givenWorkspaceGroupLabel = "Workspace";
const givenClassifierNavItemId = "classify";
const givenClassifierNavItemLabel = "Classifier";
const givenDashboardNavItemId = "dashboard";
const givenDashboardNavItemLabel = "Dashboard";

const givenSidebarGroups = [
  {
    label: givenWorkspaceGroupLabel,
    items: [
      { id: givenClassifierNavItemId, label: givenClassifierNavItemLabel, icon: "classify" as const },
      { id: givenDashboardNavItemId, label: givenDashboardNavItemLabel, icon: "dashboard" as const },
    ],
  },
];

function findRenderedNavLinkByLabel(navLinkLabel: string) {
  return screen
    .getAllByTestId(NAV_LINK_DATA_TEST_ID.CONTAINER)
    .find((node) => node.textContent?.includes(navLinkLabel));
}

describe("Sidebar", () => {
  it("renders the sidebar container with brand and group label", () => {
    // GIVEN a Sidebar with one nav group
    // WHEN we render it
    render(
      <Sidebar
        activeId={givenClassifierNavItemId}
        onNavigate={() => {}}
        groups={givenSidebarGroups}
      />,
    );

    // THEN the container, brand button, and group label are present
    expect(screen.getByTestId(DATA_TEST_ID.CONTAINER)).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.BRAND_BUTTON)).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.GROUP_LABEL)).toHaveTextContent(
      givenWorkspaceGroupLabel,
    );
  });

  it("marks the active nav item with aria-current=page", () => {
    // GIVEN a Sidebar with the dashboard nav item active
    // WHEN we render it
    render(
      <Sidebar
        activeId={givenDashboardNavItemId}
        onNavigate={() => {}}
        groups={givenSidebarGroups}
      />,
    );

    // THEN the dashboard nav link is current and the classifier one is not
    expect(
      findRenderedNavLinkByLabel(givenDashboardNavItemLabel)?.getAttribute(
        "aria-current",
      ),
    ).toBe("page");
    expect(
      findRenderedNavLinkByLabel(givenClassifierNavItemLabel)?.getAttribute(
        "aria-current",
      ),
    ).toBeNull();
  });

  it("calls onNavigate with the id when a nav item is clicked", async () => {
    // GIVEN an onNavigate spy
    const onNavigate = vi.fn();

    // AND a rendered Sidebar with the classifier nav item active
    render(
      <Sidebar
        activeId={givenClassifierNavItemId}
        onNavigate={onNavigate}
        groups={givenSidebarGroups}
      />,
    );

    // WHEN the user clicks the dashboard nav item
    await userEvent.click(
      findRenderedNavLinkByLabel(givenDashboardNavItemLabel)!,
    );

    // THEN onNavigate is called with the dashboard nav item id
    expect(onNavigate).toHaveBeenCalledWith(givenDashboardNavItemId);
  });

  it("calls onSignOut when the sign-out button is clicked", async () => {
    // GIVEN an onSignOut spy and the signed-in user
    const onSignOut = vi.fn();
    const givenUserInitials = "SM";
    const givenUserEmail = "sara@tabiya.org";

    // AND a Sidebar rendered with that user and the sign-out handler
    render(
      <Sidebar
        activeId={givenClassifierNavItemId}
        onNavigate={() => {}}
        groups={givenSidebarGroups}
        user={{ initials: givenUserInitials, label: givenUserEmail }}
        onSignOut={onSignOut}
      />,
    );

    // WHEN the user clicks the sign-out button
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.SIGN_OUT));

    // THEN onSignOut is invoked, and the user's info+avatar are visible
    expect(onSignOut).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId(DATA_TEST_ID.USER_INFO)).toHaveTextContent(
      givenUserEmail,
    );
    expect(screen.getByTestId(DATA_TEST_ID.USER_AVATAR)).toHaveTextContent(
      givenUserInitials,
    );
  });
});
