import { describe, expect, it } from "vitest";
import type {
  ClassifiedEntity,
  ClassifyEntityType,
  ClassifyMatch,
} from "@/lib/api";
import { splitSpans } from "./splitSpans";

function makeEntity(opts: {
  start: number;
  end: number;
  surface: string;
  type?: ClassifyEntityType;
  score?: number;
}): ClassifiedEntity {
  const entityType: ClassifyEntityType = opts.type ?? "skill";
  const match: ClassifyMatch =
    entityType === "occupation"
      ? {
          entity_type: "occupation",
          similarity_score: opts.score ?? 0.9,
          entity: {
            uuid: "u",
            origin_uuid: "u",
            uuid_history: [],
            preferred_label: opts.surface,
            origin_uri: "",
            alt_labels: [],
            description: "",
          },
        }
      : entityType === "qualification"
      ? {
          entity_type: "qualification",
          similarity_score: opts.score ?? 0.9,
          entity: {
            uuid: "u",
            origin_uuid: "u",
            uuid_history: [],
            preferred_label: opts.surface,
            origin_uri: "",
            alt_labels: [],
            description: "",
          },
        }
      : {
          entity_type: "skill",
          similarity_score: opts.score ?? 0.9,
          entity: {
            uuid: "u",
            origin_uuid: "u",
            uuid_history: [],
            preferred_label: opts.surface,
            origin_uri: "",
            alt_labels: [],
            description: "",
          },
        };
  return {
    entity_type: entityType,
    surface_form: opts.surface,
    span: { start: opts.start, end: opts.end },
    matches: [match],
  };
}

describe("splitSpans", () => {
  it("returns a single text segment when there are no entities", () => {
    // GIVEN plain text and zero entities
    const givenText = "Just plain text";

    // WHEN we split
    const segments = splitSpans(givenText, []);

    // THEN one text segment covering the whole string
    expect(segments).toEqual([
      { kind: "text", text: givenText, start: 0 },
    ]);
  });

  it("returns one text + one entity + one text for a centered span", () => {
    // GIVEN "Python is great" with the word "Python" tagged
    const givenText = "Python is great";
    const givenEntity = makeEntity({ start: 0, end: 6, surface: "Python" });

    // WHEN we split
    const segments = splitSpans(givenText, [givenEntity]);

    // THEN entity comes first, then trailing text
    expect(segments).toHaveLength(2);
    expect(segments[0]).toMatchObject({ kind: "ent", text: "Python", start: 0, end: 6 });
    expect(segments[1]).toEqual({ kind: "text", text: " is great", start: 6 });
  });

  it("preserves leading text when the first entity does not start at offset 0", () => {
    // GIVEN text with an entity in the middle
    const givenText = "I love Python";
    const givenEntity = makeEntity({ start: 7, end: 13, surface: "Python" });

    // WHEN we split
    const segments = splitSpans(givenText, [givenEntity]);

    // THEN leading text comes first
    expect(segments).toHaveLength(2);
    expect(segments[0]).toEqual({ kind: "text", text: "I love ", start: 0 });
    expect(segments[1]).toMatchObject({ kind: "ent", text: "Python" });
  });

  it("sorts entities by start offset before rendering", () => {
    // GIVEN two entities supplied OUT of order
    const givenText = "Python and SQL";
    const givenPythonEntity = makeEntity({ start: 0, end: 6, surface: "Python" });
    const givenSqlEntity = makeEntity({ start: 11, end: 14, surface: "SQL" });

    // WHEN we split with SQL first in the input
    const segments = splitSpans(givenText, [givenSqlEntity, givenPythonEntity]);

    // THEN the segments are emitted in text order
    const entSegments = segments.filter((s) => s.kind === "ent");
    expect(entSegments).toHaveLength(2);
    expect(entSegments[0]).toMatchObject({ text: "Python", start: 0 });
    expect(entSegments[1]).toMatchObject({ text: "SQL", start: 11 });
  });

  it("preserves the original index even when entities are reordered internally", () => {
    // GIVEN two entities supplied OUT of text order
    const givenText = "Python and SQL";
    const givenSqlEntity = makeEntity({ start: 11, end: 14, surface: "SQL" });
    const givenPythonEntity = makeEntity({ start: 0, end: 6, surface: "Python" });

    // WHEN we split with SQL at input index 0 and Python at input index 1
    const segments = splitSpans(givenText, [givenSqlEntity, givenPythonEntity]);

    // THEN the entity segments expose the original input index
    const entSegments = segments.filter((s) => s.kind === "ent") as Array<{ entityIndex: number }>;
    expect(entSegments[0].entityIndex).toBe(1); // Python was input index 1
    expect(entSegments[1].entityIndex).toBe(0); // SQL was input index 0
  });

  it("drops the lower-scoring entity when two spans overlap", () => {
    // GIVEN two overlapping entities, one scoring higher
    const givenText = "data scientist role";
    const givenHigh = makeEntity({
      start: 0,
      end: 14,
      surface: "data scientist",
      score: 0.95,
    });
    const givenLow = makeEntity({
      start: 5,
      end: 14,
      surface: "scientist",
      score: 0.6,
    });

    // WHEN we split
    const segments = splitSpans(givenText, [givenHigh, givenLow]);

    // THEN only the higher-scoring entity is kept
    const entSegments = segments.filter((s) => s.kind === "ent");
    expect(entSegments).toHaveLength(1);
    expect(entSegments[0]).toMatchObject({
      text: "data scientist",
      start: 0,
      end: 14,
    });
  });

  it("treats adjacent (non-overlapping) spans as both kept", () => {
    // GIVEN two spans where the second starts exactly where the first ends
    const givenText = "PythonSQL";
    const givenPython = makeEntity({ start: 0, end: 6, surface: "Python" });
    const givenSql = makeEntity({ start: 6, end: 9, surface: "SQL" });

    // WHEN we split
    const segments = splitSpans(givenText, [givenPython, givenSql]);

    // THEN both entities render with no gap text between them
    const kinds = segments.map((s) => s.kind);
    expect(kinds).toEqual(["ent", "ent"]);
  });

  it("drops zero-length spans (start == end)", () => {
    // GIVEN a zero-length span
    const givenText = "valid text";
    const givenBad = makeEntity({ start: 5, end: 5, surface: "" });

    // WHEN we split
    const segments = splitSpans(givenText, [givenBad]);

    // THEN no entity is emitted, just plain text
    expect(segments).toEqual([{ kind: "text", text: "valid text", start: 0 }]);
  });

  it("drops spans that extend past the end of the text", () => {
    // GIVEN a span whose end is beyond the text length
    const givenText = "short";
    const givenBad = makeEntity({ start: 0, end: 100, surface: "short" });

    // WHEN we split
    const segments = splitSpans(givenText, [givenBad]);

    // THEN no entity is emitted
    expect(segments).toEqual([{ kind: "text", text: "short", start: 0 }]);
  });

  it("drops spans with negative start offsets", () => {
    // GIVEN a span with a negative start
    const givenText = "hello";
    const givenBad = makeEntity({ start: -1, end: 3, surface: "hel" });

    // WHEN we split
    const segments = splitSpans(givenText, [givenBad]);

    // THEN no entity is emitted
    expect(segments).toEqual([{ kind: "text", text: "hello", start: 0 }]);
  });

  it("emits a trailing text segment when entities don't reach the end", () => {
    // GIVEN an entity that ends before the text
    const givenText = "Python here";
    const givenEntity = makeEntity({ start: 0, end: 6, surface: "Python" });

    // WHEN we split
    const segments = splitSpans(givenText, [givenEntity]);

    // THEN trailing text appears
    expect(segments[segments.length - 1]).toEqual({
      kind: "text",
      text: " here",
      start: 6,
    });
  });

  it("preserves whitespace and punctuation in gap text", () => {
    // GIVEN entities separated by punctuation
    const givenText = "uses Python, SQL.";
    const givenPython = makeEntity({ start: 5, end: 11, surface: "Python" });
    const givenSql = makeEntity({ start: 13, end: 16, surface: "SQL" });

    // WHEN we split
    const segments = splitSpans(givenText, [givenPython, givenSql]);

    // THEN the comma+space stays as plain text between the two entities
    const textSegments = segments.filter((s) => s.kind === "text");
    expect(textSegments.map((s) => s.text)).toEqual(["uses ", ", ", "."]);
  });
});
