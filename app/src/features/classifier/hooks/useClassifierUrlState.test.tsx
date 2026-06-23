import { describe, expect, it } from "vitest";
import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter, useLocation } from "react-router-dom";
import {
  MIN_SIMILARITY_DEFAULT,
  TOP_K_DEFAULT,
  useClassifierUrlState,
} from "./useClassifierUrlState";

function wrapAtUrl(initialUrl: string) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <MemoryRouter initialEntries={[initialUrl]}>{children}</MemoryRouter>
    );
  };
}

/** Reads the current location.search inside the same router. */
function useCurrentSearch() {
  return useLocation().search;
}

describe("useClassifierUrlState", () => {
  it("returns the supplied defaults when the URL has no params", () => {
    // GIVEN a URL with no search params
    // WHEN we mount the hook
    const { result } = renderHook(() => useClassifierUrlState(), {
      wrapper: wrapAtUrl("/classifier"),
    });

    // THEN the defaults surface
    expect(result.current.topK).toBe(TOP_K_DEFAULT);
    expect(result.current.minSimilarity).toBe(MIN_SIMILARITY_DEFAULT);
  });

  it("parses well-formed URL params", () => {
    // GIVEN a URL with top_k=8 and min_sim=0.4
    // WHEN we mount
    const { result } = renderHook(() => useClassifierUrlState(), {
      wrapper: wrapAtUrl("/classifier?top_k=8&min_sim=0.4"),
    });

    // THEN the parsed values surface
    expect(result.current.topK).toBe(8);
    expect(result.current.minSimilarity).toBe(0.4);
  });

  it("clamps top_k above the max back to 50", () => {
    // GIVEN an out-of-range top_k
    // WHEN we mount
    const { result } = renderHook(() => useClassifierUrlState(), {
      wrapper: wrapAtUrl("/classifier?top_k=9999"),
    });

    // THEN the value is clamped to TOP_K_MAX (50)
    expect(result.current.topK).toBe(50);
  });

  it("falls back to defaults when the params are unparseable", () => {
    // GIVEN garbage values
    // WHEN we mount
    const { result } = renderHook(() => useClassifierUrlState(), {
      wrapper: wrapAtUrl("/classifier?top_k=banana&min_sim=peach"),
    });

    // THEN the defaults surface
    expect(result.current.topK).toBe(TOP_K_DEFAULT);
    expect(result.current.minSimilarity).toBe(MIN_SIMILARITY_DEFAULT);
  });

  it("setTopK updates the URL and the surfaced value", () => {
    // GIVEN a hook at the default URL with the location reader sharing it
    const { result } = renderHook(
      () => {
        const state = useClassifierUrlState();
        const search = useCurrentSearch();
        return { state, search };
      },
      { wrapper: wrapAtUrl("/classifier") },
    );

    // WHEN we call setTopK(7)
    act(() => result.current.state.setTopK(7));

    // THEN both the surfaced value AND the URL reflect the new state
    expect(result.current.state.topK).toBe(7);
    expect(result.current.search).toContain("top_k=7");
  });

  it("setMinSimilarity clamps and formats the URL value", () => {
    // GIVEN the default URL
    const { result } = renderHook(
      () => {
        const state = useClassifierUrlState();
        const search = useCurrentSearch();
        return { state, search };
      },
      { wrapper: wrapAtUrl("/classifier") },
    );

    // WHEN we call setMinSimilarity(2.5) — above the cap
    act(() => result.current.state.setMinSimilarity(2.5));

    // THEN the value is clamped to 1.00 and the URL is formatted to 2 decimals
    expect(result.current.state.minSimilarity).toBe(1);
    expect(result.current.search).toContain("min_sim=1.00");
  });
});
