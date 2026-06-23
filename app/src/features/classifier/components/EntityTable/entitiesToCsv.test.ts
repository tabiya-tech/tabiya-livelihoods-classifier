import { describe, expect, it } from "vitest";
import { fixtureClassifyEntities } from "@/mocks/fixtures/classify";
import type { ClassifiedEntity } from "@/lib/api";
import { CSV_COLUMNS, entitiesToCsv } from "./entitiesToCsv";

describe("entitiesToCsv", () => {
  it("emits exactly the canonical header on the first line", () => {
    // GIVEN an empty entities list
    const expectedHeader = CSV_COLUMNS.join(",");

    // WHEN we serialize
    const csv = entitiesToCsv([]);

    // THEN only the header is emitted
    expect(csv).toBe(expectedHeader);
  });

  it("emits one row per (entity, match) pair", () => {
    // GIVEN the fixture (5 entities, total 7 matches across them)
    const expectedDataRows = fixtureClassifyEntities.reduce(
      (total, entity) => total + Math.max(entity.matches.length, 1),
      0,
    );

    // WHEN we serialize
    const csv = entitiesToCsv(fixtureClassifyEntities);

    // THEN row count = data rows + 1 header line
    const lineCount = csv.split("\n").length;
    expect(lineCount).toBe(expectedDataRows + 1);
  });

  it("emits a row with blank match columns when an entity has zero matches", () => {
    // GIVEN an entity with no matches
    const givenEntity: ClassifiedEntity = {
      ...fixtureClassifyEntities[0],
      matches: [],
    };

    // WHEN we serialize
    const csv = entitiesToCsv([givenEntity]);
    const dataLine = csv.split("\n")[1];

    // THEN the match_rank / match_label / match_score / match_uri cells are empty
    const cells = dataLine.split(",");
    expect(cells[0]).toBe(givenEntity.entity_type);
    expect(cells[1]).toBe(givenEntity.surface_form);
    // match_rank, match_label, match_score, match_uri are positions 4..7
    expect(cells[4]).toBe("");
    expect(cells[5]).toBe("");
    expect(cells[6]).toBe("");
    expect(cells[7]).toBe("");
  });

  it("quotes and escapes cells that contain commas or quotes", () => {
    // GIVEN an entity whose label has a comma and a quote
    const givenEntity: ClassifiedEntity = {
      ...fixtureClassifyEntities[0],
      surface_form: 'oddly, "quoted" form',
    };

    // WHEN we serialize
    const csv = entitiesToCsv([givenEntity]);
    const dataLine = csv.split("\n")[1];

    // THEN the surface_form column is wrapped in quotes and inner quotes are doubled
    expect(dataLine).toContain(`"oddly, ""quoted"" form"`);
  });

  it("ranks matches starting at 1 in their backend order", () => {
    // GIVEN the first fixture entity has two matches
    const givenEntity = fixtureClassifyEntities[0];
    expect(givenEntity.matches.length).toBeGreaterThanOrEqual(2);

    // WHEN we serialize
    const csv = entitiesToCsv([givenEntity]);
    const dataLines = csv.split("\n").slice(1);

    // THEN match_rank reads "1", "2", … in that order
    const ranks = dataLines.map((line) => line.split(",")[4]);
    expect(ranks.slice(0, givenEntity.matches.length)).toEqual(
      givenEntity.matches.map((_match, index) => String(index + 1)),
    );
  });
});
