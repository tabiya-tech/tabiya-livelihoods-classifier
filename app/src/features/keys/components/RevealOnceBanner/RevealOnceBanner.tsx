/**
 * Sticky banner that surfaces a just-issued plaintext API key with a Copy
 * button. The key is unrecoverable after the user dismisses the banner.
 *
 * The page hides this when useCreateApiKey.justIssued is null.
 */

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Button, Icon } from "@/components";
import { mergeClassNames } from "@/lib/mergeClassNames";

const uniqueId = "2d8b4f7c-1e9a-4d5b-8c7f-3a6e1d4b9c2f";

export const DATA_TEST_ID = {
  CONTAINER: `reveal-once-banner-container-${uniqueId}`,
  TITLE: `reveal-once-banner-title-${uniqueId}`,
  DESCRIPTION: `reveal-once-banner-description-${uniqueId}`,
  KEY_VALUE: `reveal-once-banner-key-value-${uniqueId}`,
  COPY_BUTTON: `reveal-once-banner-copy-button-${uniqueId}`,
  DISMISS_BUTTON: `reveal-once-banner-dismiss-button-${uniqueId}`,
};

export interface RevealOnceBannerProps {
  /** The plaintext key. Required — there is no other way to show it later. */
  apiKey: string;
  onDismiss: () => void;
  className?: string;
}

export function RevealOnceBanner({
  apiKey,
  onDismiss,
  className,
}: RevealOnceBannerProps) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(apiKey);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // navigator.clipboard can be unavailable in tests/older browsers — swallow.
    }
  }

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      role="alert"
      aria-live="assertive"
      className={mergeClassNames(
        "flex flex-col gap-3 rounded-md border border-lime-600 bg-lime/15 px-4 py-3.5",
        className,
      )}
    >
      <div className="flex items-start gap-2">
        <Icon name="check" className="mt-0.5 text-teal" />
        <div className="flex-1">
          <div
            data-testid={DATA_TEST_ID.TITLE}
            className="font-mono text-xs font-medium text-navy"
          >
            {t("keys.revealBanner.title")}
          </div>
          <div
            data-testid={DATA_TEST_ID.DESCRIPTION}
            className="mt-0.5 text-xs leading-snug text-muted"
          >
            {t("keys.revealBanner.description")}
          </div>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <code
          data-testid={DATA_TEST_ID.KEY_VALUE}
          className="flex-1 select-all overflow-x-auto rounded border border-line bg-paper px-2.5 py-1.5 font-mono text-xs text-navy"
        >
          {apiKey}
        </code>
        <Button
          size="sm"
          variant="default"
          onClick={() => {
            void handleCopy();
          }}
          leading={<Icon name="copy" />}
          data-testid={DATA_TEST_ID.COPY_BUTTON}
        >
          {copied ? t("common.buttons.copied") : t("common.buttons.copy")}
        </Button>
        <Button
          size="sm"
          variant="primary"
          onClick={onDismiss}
          data-testid={DATA_TEST_ID.DISMISS_BUTTON}
        >
          {t("keys.revealBanner.dismiss")}
        </Button>
      </div>
    </div>
  );
}
