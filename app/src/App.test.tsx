import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

const GIVEN_APP_ROUTER_MARKER_TEST_ID = "given-app-router-marker";

// Stub the router's children so this test stays focused on App composition
// rather than the inner feature pages (those have their own tests).
vi.mock("./routes/AppRouter", () => ({
  AppRouter: () => (
    <div data-testid={GIVEN_APP_ROUTER_MARKER_TEST_ID}>router</div>
  ),
}));

import { App } from "./App";

describe("App", () => {
  it("renders the AppRouter", () => {
    // GIVEN the App component
    // WHEN it is rendered
    render(<App />);

    // THEN the AppRouter is mounted
    expect(
      screen.getByTestId(GIVEN_APP_ROUTER_MARKER_TEST_ID),
    ).toBeInTheDocument();
  });
});
