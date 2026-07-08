/**
 * Read-only JSON dump of the classify response. Two actions:
 *   - Copy to clipboard
 *   - Download as `.json` file
 *
 * Uses CodeBlock for the rendered body (which already supplies copy + token
 * coloring) but adds a Download button for one-click export.
 */

import { useTranslation } from "react-i18next";
import { Button, Icon } from "@/components";
import { mergeClassNames } from "@/lib/mergeClassNames";

const uniqueId = "5d1a3f8c-7e9b-4c2d-8a4f-6b3e2d5c1f8a";

export const DATA_TEST_ID = {
  CONTAINER: `json-view-container-${uniqueId}`,
  PRE: `json-view-pre-${uniqueId}`,
  DOWNLOAD_BUTTON: `json-view-download-button-${uniqueId}`,
};

export interface JsonViewProps {
  /** Anything JSON-serializable. */
  value: unknown;
  /** Filename for the download (without extension). */
  filename?: string;
  className?: string;
}

export function JsonView({
  value,
  filename = "classification",
  className,
}: JsonViewProps) {
  const { t } = useTranslation();
  const json = JSON.stringify(value, null, 2);

  function handleDownload() {
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${filename}.json`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames("flex min-h-0 flex-col gap-2", className)}
    >
      <div className="flex justify-end">
        <Button
          size="sm"
          variant="default"
          leading={<Icon name="download" />}
          onClick={handleDownload}
          data-testid={DATA_TEST_ID.DOWNLOAD_BUTTON}
        >
          {t("classifier.results.downloadJson")}
        </Button>
      </div>
      <pre
        data-testid={DATA_TEST_ID.PRE}
        className="m-0 min-h-0 flex-1 overflow-auto rounded-md border border-line bg-ink p-4 font-mono text-[12px] leading-relaxed text-lime"
      >
        {json}
      </pre>
    </div>
  );
}
