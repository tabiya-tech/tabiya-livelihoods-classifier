/**
 * Sticky save bar for the Configuration page. Slides in from below when
 * there are unsaved changes (or when a save is in flight / just succeeded)
 * and hides itself when the draft is clean again.
 *
 * The state lives in `useUserConfiguration` on the page. This component is
 * pure presentation — it reads `saveStatus` + `isDirty` and surfaces the
 * Discard / Save buttons.
 */

import { AnimatePresence, motion } from "framer-motion";
import { useTranslation } from "react-i18next";
import type { TranslationKey } from "@/react-i18next";
import { Button, Icon } from "@/components";
import type { ConfigurationSaveStatus } from "../../hooks/useUserConfiguration";

const uniqueId = "f3a8d6e1-2b4c-4f5a-8e9d-1c7b3a6f2d4e";

export const DATA_TEST_ID = {
  CONTAINER: `save-bar-container-${uniqueId}`,
  TITLE: `save-bar-title-${uniqueId}`,
  HELP: `save-bar-help-${uniqueId}`,
  DISCARD_BUTTON: `save-bar-discard-button-${uniqueId}`,
  SAVE_BUTTON: `save-bar-save-button-${uniqueId}`,
  SAVED_INDICATOR: `save-bar-saved-indicator-${uniqueId}`,
};

export interface SaveBarProps {
  /** True when draft and saved diverge. Drives whether the bar is visible. */
  isDirty: boolean;
  /** Current save lifecycle status. */
  saveStatus: ConfigurationSaveStatus;
  onSave: () => void;
  onDiscard: () => void;
  /** Horizontal offset from the left edge of the viewport (sidebar width). */
  leftOffset?: number | string;
}

interface SaveBarCopyKeys {
  titleKey: TranslationKey;
  helpKey: TranslationKey;
}

function deriveCopyKeys(
  saveStatus: ConfigurationSaveStatus,
  isDirty: boolean,
): SaveBarCopyKeys | null {
  if (saveStatus === "saving") {
    return {
      titleKey: "configuration.saveBar.savingTitle",
      helpKey: "configuration.saveBar.savingHelp",
    };
  }
  if (saveStatus === "saved") {
    return {
      titleKey: "configuration.saveBar.savedTitle",
      helpKey: "configuration.saveBar.savedHelp",
    };
  }
  if (isDirty) {
    return {
      titleKey: "configuration.saveBar.dirtyTitle",
      helpKey: "configuration.saveBar.dirtyHelp",
    };
  }
  return null;
}

export function SaveBar({
  isDirty,
  saveStatus,
  onSave,
  onDiscard,
  leftOffset = 232,
}: SaveBarProps) {
  const { t } = useTranslation();
  const copyKeys = deriveCopyKeys(saveStatus, isDirty);
  const isVisible = copyKeys !== null;
  const isSaving = saveStatus === "saving";
  const isSaved = saveStatus === "saved";

  return (
    <AnimatePresence>
      {isVisible && copyKeys && (
        <motion.div
          data-testid={DATA_TEST_ID.CONTAINER}
          role="status"
          aria-live="polite"
          initial={{ y: "100%" }}
          animate={{ y: 0 }}
          exit={{ y: "100%" }}
          transition={{ duration: 0.2, ease: [0.2, 0.8, 0.2, 1] }}
          style={{ left: leftOffset }}
          className="fixed bottom-0 right-0 z-30 flex items-center justify-between gap-4 border-t border-line bg-cream px-8 py-3.5 shadow-[0_-4px_16px_rgba(0,0,0,0.04)]"
        >
          <div>
            <div
              data-testid={DATA_TEST_ID.TITLE}
              className="font-mono text-xs font-medium text-navy"
            >
              {t(copyKeys.titleKey)}
            </div>
            <div
              data-testid={DATA_TEST_ID.HELP}
              className="mt-0.5 text-xs leading-snug text-muted"
            >
              {t(copyKeys.helpKey)}
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isSaved && (
              <span
                data-testid={DATA_TEST_ID.SAVED_INDICATOR}
                className="inline-flex items-center gap-1 font-mono text-xs text-teal"
              >
                <Icon name="check" />
                {t("configuration.saveBar.savedIndicator")}
              </span>
            )}
            <Button
              variant="ghost"
              onClick={onDiscard}
              disabled={!isDirty || isSaving}
              data-testid={DATA_TEST_ID.DISCARD_BUTTON}
            >
              {t("common.buttons.discard")}
            </Button>
            <Button
              variant="primary"
              onClick={onSave}
              loading={isSaving}
              disabled={!isDirty}
              data-testid={DATA_TEST_ID.SAVE_BUTTON}
            >
              {t("common.buttons.save")}
            </Button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
