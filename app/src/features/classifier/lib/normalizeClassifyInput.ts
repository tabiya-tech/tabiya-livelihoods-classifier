/**
 * Normalise free-form text before sending it to the Classifier backend.
 *
 * Why: the Tabiya NER model (`tabiya/roberta-base-job-ner`) was trained on
 * prose-style job descriptions. When a user uploads a job spec laid out as
 * a form ("Job title\nStatistician\nDepartment\n"), the model reads single
 * tokens on isolated lines as form scaffolding and skips them — so the
 * occupation never makes it to NEL.
 *
 * This normalisation rewrites single newlines between alphanumeric lines as
 * spaces, while preserving real paragraph breaks (two-or-more consecutive
 * newlines). The result is text that reads like prose to the model without
 * losing the structure the user intended.
 *
 * NOTE: the normalised text is the source of truth for span offsets that
 * the backend returns, so it must also become the text the UI renders
 * inline highlights against.
 */

const WHITESPACE_TRIM = /^[\t ]+|[\t ]+$/gm;
const THREE_OR_MORE_NEWLINES = /\n{3,}/g;
const SINGLE_NEWLINE_BETWEEN_CONTENT = /([^\n])\n(?!\n)([^\n])/g;
/** Any run of tab(s) or 2+ spaces → one space. Single inter-word spaces are left alone. */
const INLINE_WHITESPACE_RUN = /\t+| {2,}/g;

export function normalizeClassifyInput(raw: string): string {
  // Normalise CR/CRLF → LF so the rest of the pipeline only deals with \n.
  let text = raw.replace(/\r\n?/g, "\n");

  // Trim trailing/leading whitespace from every line — strips invisible
  // tabs that would otherwise survive the newline-collapse below.
  text = text.replace(WHITESPACE_TRIM, "");

  // Collapse runs of 3+ newlines to exactly 2 (paragraph break). This must
  // run BEFORE the single-newline join so the regex below doesn't have to
  // worry about 3+ runs masquerading as paragraph breaks.
  text = text.replace(THREE_OR_MORE_NEWLINES, "\n\n");

  // Join single newlines between non-blank lines with a space. This is the
  // step that turns form-style layouts back into prose. We use a function
  // replacement so the regex's two captured characters are reinserted
  // around the new space.
  text = text.replace(SINGLE_NEWLINE_BETWEEN_CONTENT, "$1 $2");

  // Collapse any tab or run of 2+ spaces to a single space.
  text = text.replace(INLINE_WHITESPACE_RUN, " ");

  return text.trim();
}
