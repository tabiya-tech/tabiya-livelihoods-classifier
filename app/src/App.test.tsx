import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { App } from "./App";

describe("App", () => {
  it("renders the bootstrap message", () => {
    // GIVEN the App component
    // WHEN it is rendered
    render(<App />);

    // THEN the scaffolding-ready message is visible to the user
    expect(screen.getByText(/scaffolding ready/i)).toBeInTheDocument();
  });
});
