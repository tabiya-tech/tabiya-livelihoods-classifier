/**
 * Right-side drawer surfaced when the user clicks an entity. Shows the
 * surface form, entity type, and the full list of ESCO matches with score
 * bars + a link out to the taxonomy URI.
 *
 * Pure presentation. The page owns which entity is currently selected.
 */

import { useTranslation } from "react-i18next";
import { Drawer, Icon, ScoreBar, Tag } from "@/components";
import type {
  ClassifiedEntity,
  ClassifyMatch,
  ClassifyOccupationMatch,
  ClassifyQualificationMatch,
  ClassifySkillMatch,
} from "@/lib/api";
import { mergeClassNames } from "@/lib/mergeClassNames";
import { EntitySwatch } from "../EntitySwatch/EntitySwatch";

const uniqueId = "3a8e5d2c-7f1b-4c9d-8e4f-1b6a3d8c5e2f";

export const DATA_TEST_ID = {
  HEADER: `entity-detail-drawer-header-${uniqueId}`,
  SURFACE_FORM: `entity-detail-drawer-surface-form-${uniqueId}`,
  SPAN: `entity-detail-drawer-span-${uniqueId}`,
  MATCH_ROW: `entity-detail-drawer-match-row-${uniqueId}`,
  MATCH_LABEL: `entity-detail-drawer-match-label-${uniqueId}`,
  MATCH_SCORE: `entity-detail-drawer-match-score-${uniqueId}`,
  MATCH_LINK: `entity-detail-drawer-match-link-${uniqueId}`,
  MATCH_DESCRIPTION: `entity-detail-drawer-match-description-${uniqueId}`,
  EMPTY_NOTE: `entity-detail-drawer-empty-note-${uniqueId}`,
};

export interface EntityDetailDrawerProps {
  open: boolean;
  /** The entity to detail. Required when open; null is treated as closed. */
  entity: ClassifiedEntity | null;
  onClose: () => void;
}

function isOccupationMatch(
  match: ClassifyMatch,
): match is ClassifyOccupationMatch {
  return match.entity_type === "occupation";
}

function isSkillMatch(match: ClassifyMatch): match is ClassifySkillMatch {
  return match.entity_type === "skill";
}

function isQualificationMatch(
  match: ClassifyMatch,
): match is ClassifyQualificationMatch {
  return match.entity_type === "qualification";
}

export function EntityDetailDrawer({
  open,
  entity,
  onClose,
}: EntityDetailDrawerProps) {
  const { t } = useTranslation();
  // Pass open=false when there's no entity so the panel slides out cleanly.
  const isVisible = open && entity !== null;

  return (
    <Drawer
      open={isVisible}
      onClose={onClose}
      eyebrow={
        entity ? (
          <span className="inline-flex items-center gap-2">
            <EntitySwatch entityType={entity.entity_type} />
            {t(`classifier.entityTypeFilter.types.${entity.entity_type}`)}
          </span>
        ) : undefined
      }
      title={entity?.surface_form ?? ""}
      description={
        entity ? (
          <span
            data-testid={DATA_TEST_ID.SPAN}
            className="font-mono text-[11px] text-muted-2"
          >
            {t("classifier.entityDetail.spanLabel", {
              start: entity.span.start,
              end: entity.span.end,
            })}
          </span>
        ) : undefined
      }
      width={520}
    >
      {entity && (
        <div className="flex flex-col gap-3">
          <h3 className="m-0 font-mono text-[11px] uppercase tracking-[0.08em] text-muted">
            {t("classifier.entityDetail.matchesHeading", {
              count: entity.matches.length,
            })}
          </h3>
          {entity.matches.length === 0 ? (
            <p
              data-testid={DATA_TEST_ID.EMPTY_NOTE}
              className="m-0 text-xs italic text-muted"
            >
              {t("classifier.results.noMatches")}
            </p>
          ) : (
            entity.matches.map((match, matchIndex) => (
              <MatchCard
                key={`${matchIndex}-${match.entity.uuid}`}
                match={match}
              />
            ))
          )}
        </div>
      )}
    </Drawer>
  );
}

interface MatchCardProps {
  match: ClassifyMatch;
}

function MatchCard({ match }: MatchCardProps) {
  const { t } = useTranslation();
  const scorePercent = `${(match.similarity_score * 100).toFixed(0)}%`;

  return (
    <article
      data-testid={DATA_TEST_ID.MATCH_ROW}
      className="flex min-w-0 flex-col gap-2 overflow-hidden rounded-md border border-line bg-paper px-3.5 py-3"
    >
      <header className="flex items-baseline justify-between gap-3">
        <span
          data-testid={DATA_TEST_ID.MATCH_LABEL}
          className="min-w-0 break-words font-mono text-[13px] font-medium text-navy"
        >
          {match.entity.preferred_label}
        </span>
        <span
          data-testid={DATA_TEST_ID.MATCH_SCORE}
          className="font-mono text-[11px] text-muted"
        >
          {scorePercent}
        </span>
      </header>
      <ScoreBar score={match.similarity_score} />
      {match.entity.description && (
        <p
          data-testid={DATA_TEST_ID.MATCH_DESCRIPTION}
          className="m-0 text-xs leading-relaxed text-muted"
        >
          {match.entity.description}
        </p>
      )}
      <MatchAttributes match={match} />
      {match.entity.origin_uri && (
        <a
          data-testid={DATA_TEST_ID.MATCH_LINK}
          href={match.entity.origin_uri}
          target="_blank"
          rel="noreferrer"
          className={mergeClassNames(
            "inline-flex items-center gap-1.5 self-start font-mono text-[11px] text-navy",
            "underline decoration-line-strong underline-offset-2 hover:decoration-navy",
          )}
        >
          {t("classifier.entityDetail.openInTaxonomy")}
          <Icon name="external" size={12} />
        </a>
      )}
    </article>
  );
}

function MatchAttributes({ match }: MatchCardProps) {
  const { t } = useTranslation();
  const tags: Array<{ label: string }> = [];
  if (isOccupationMatch(match) && match.entity.esco_code) {
    tags.push({
      label: t("classifier.entityDetail.attrEscoCode", {
        value: match.entity.esco_code,
      }),
    });
  }
  if (isSkillMatch(match)) {
    if (match.entity.skill_type) {
      tags.push({
        label: t("classifier.entityDetail.attrSkillType", {
          value: match.entity.skill_type,
        }),
      });
    }
    if (match.entity.reuse_level) {
      tags.push({
        label: t("classifier.entityDetail.attrReuseLevel", {
          value: match.entity.reuse_level,
        }),
      });
    }
  }
  if (isQualificationMatch(match)) {
    if (match.entity.eqf_level) {
      tags.push({
        label: t("classifier.entityDetail.attrEqfLevel", {
          value: match.entity.eqf_level,
        }),
      });
    }
    if (match.entity.country) {
      tags.push({
        label: t("classifier.entityDetail.attrCountry", {
          value: match.entity.country,
        }),
      });
    }
  }
  const altLabelsPreview =
    match.entity.alt_labels.length > 0
      ? match.entity.alt_labels.slice(0, 3).join(", ")
      : null;
  if (tags.length === 0 && altLabelsPreview === null) return null;
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      {tags.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {tags.map((tag) => (
            <Tag key={tag.label} size="sm">
              {tag.label}
            </Tag>
          ))}
        </div>
      )}
      {altLabelsPreview !== null && (
        <p className="m-0 break-words font-mono text-[11px] leading-relaxed text-muted-2">
          {t("classifier.entityDetail.attrAltLabels", {
            value: altLabelsPreview,
          })}
        </p>
      )}
    </div>
  );
}
