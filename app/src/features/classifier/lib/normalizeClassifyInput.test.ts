import { describe, expect, it } from "vitest";
import { normalizeClassifyInput } from "./normalizeClassifyInput";

describe("normalizeClassifyInput", () => {
  it("leaves prose with no newlines untouched", () => {
    // GIVEN prose text
    const givenInput = "Senior data scientist with Python and SQL.";

    // WHEN normalised
    const normalised = normalizeClassifyInput(givenInput);

    // THEN it's returned verbatim
    expect(normalised).toBe(givenInput);
  });

  it("collapses single newlines between content lines to a single space", () => {
    // GIVEN form-style layout that suppresses NER
    const givenInput = "Job title\nStatistician\nDepartment";

    // WHEN normalised
    const normalised = normalizeClassifyInput(givenInput);

    // THEN the lines join into a single sentence
    expect(normalised).toBe("Job title Statistician Department");
  });

  it("preserves paragraph breaks (two consecutive newlines)", () => {
    // GIVEN paragraphs separated by blank lines
    const givenInput = "First paragraph.\n\nSecond paragraph.";

    // WHEN normalised
    const normalised = normalizeClassifyInput(givenInput);

    // THEN the paragraph break survives
    expect(normalised).toBe("First paragraph.\n\nSecond paragraph.");
  });

  it("collapses runs of 3+ newlines down to exactly two", () => {
    // GIVEN overzealous blank lines
    const givenInput = "Para A.\n\n\n\n\nPara B.";

    // WHEN normalised
    const normalised = normalizeClassifyInput(givenInput);

    // THEN paragraphs are separated by a single blank line
    expect(normalised).toBe("Para A.\n\nPara B.");
  });

  it("collapses single newlines inside a paragraph but keeps the next paragraph break", () => {
    // GIVEN a mixed layout
    const givenInput = "Job title\nStatistician\n\nResponsibilities\nLead a team.";

    // WHEN normalised
    const normalised = normalizeClassifyInput(givenInput);

    // THEN each "paragraph" becomes prose; the paragraph break is intact
    expect(normalised).toBe(
      "Job title Statistician\n\nResponsibilities Lead a team.",
    );
  });

  it("trims per-line tab/space padding before collapsing", () => {
    // GIVEN lines with stray indentation
    const givenInput = "  Job title  \n  Statistician  \n";

    // WHEN normalised
    const normalised = normalizeClassifyInput(givenInput);

    // THEN no stray spaces survive
    expect(normalised).toBe("Job title Statistician");
  });

  it("normalises CRLF and bare CR to LF", () => {
    // GIVEN Windows-style line endings
    const givenInput = "Para A.\r\nPara A continued.\r\rPara B.";

    // WHEN normalised
    const normalised = normalizeClassifyInput(givenInput);

    // THEN CR / CRLF are gone, single-newline join applied, double-CR becomes paragraph break
    expect(normalised).toBe("Para A. Para A continued.\n\nPara B.");
  });

  it("collapses multiple in-line spaces or tabs to one", () => {
    // GIVEN double-tabbed or double-spaced text
    const givenInput = "Lead   a\tteam   of  experts.";

    // WHEN normalised
    const normalised = normalizeClassifyInput(givenInput);

    // THEN exactly one space sits between each word
    expect(normalised).toBe("Lead a team of experts.");
  });

  it("strips leading and trailing whitespace from the whole string", () => {
    // GIVEN text with leading + trailing whitespace
    const givenInput = "\n\n  Job title\nStatistician  \n\n";

    // WHEN normalised
    const normalised = normalizeClassifyInput(givenInput);

    // THEN the result has no surrounding whitespace
    expect(normalised).toBe("Job title Statistician");
  });

  it("returns an empty string for whitespace-only input", () => {
    // GIVEN only whitespace
    const givenInput = "  \n\n \t\n";

    // WHEN normalised
    const normalised = normalizeClassifyInput(givenInput);

    // THEN the result is empty
    expect(normalised).toBe("");
  });
});
