import { useEffect, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";

const uniqueId = "3b992191-34cd-406b-bf12-5ab1b1729d2b";

export const DATA_TEST_ID = {
  BACKDROP: `drawer-backdrop-${uniqueId}`,
  PANEL: `drawer-panel-${uniqueId}`,
  HEADER: `drawer-header-${uniqueId}`,
  EYEBROW: `drawer-eyebrow-${uniqueId}`,
  TITLE: `drawer-title-${uniqueId}`,
  DESCRIPTION: `drawer-description-${uniqueId}`,
  BODY: `drawer-body-${uniqueId}`,
  FOOTER: `drawer-footer-${uniqueId}`,
};

export interface DrawerProps {
  open: boolean;
  onClose: () => void;
  title?: ReactNode;
  description?: ReactNode;
  /** Visual eyebrow above the title (e.g. entity type name). */
  eyebrow?: ReactNode;
  /** Sticky footer slot. */
  footer?: ReactNode;
  /** Width of the slide-in panel. Defaults to 480px. */
  width?: number | string;
  /** Closes the drawer when the backdrop is clicked. Defaults to true. */
  closeOnBackdropClick?: boolean;
  className?: string;
  children?: ReactNode;
}

const backdropVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1 },
};

const panelVariants = {
  hidden: { x: 24, opacity: 0 },
  visible: { x: 0, opacity: 1 },
};

export function Drawer({
  open,
  onClose,
  title,
  description,
  eyebrow,
  footer,
  width = 480,
  closeOnBackdropClick = true,
  className,
  children,
}: DrawerProps) {
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    panelRef.current?.focus();
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  return createPortal(
    <AnimatePresence>
      {open && (
        <motion.div
          role="presentation"
          data-testid={DATA_TEST_ID.BACKDROP}
          onMouseDown={(event) => {
            if (closeOnBackdropClick && event.target === event.currentTarget) {
              onClose();
            }
          }}
          className="fixed inset-0 z-[150] flex items-end justify-end bg-ink/35"
          variants={backdropVariants}
          initial="hidden"
          animate="visible"
          exit="hidden"
          transition={{ duration: 0.16, ease: "easeOut" }}
        >
          <motion.div
            ref={panelRef}
            role="dialog"
            data-testid={DATA_TEST_ID.PANEL}
            aria-modal="true"
            aria-labelledby={title ? DATA_TEST_ID.TITLE : undefined}
            tabIndex={-1}
            style={{ width }}
            className={mergeClassNames(
              "flex h-full max-w-full flex-col border-l border-line bg-paper outline-none",
              className,
            )}
            variants={panelVariants}
            initial="hidden"
            animate="visible"
            exit="hidden"
            transition={{ duration: 0.22, ease: [0.2, 0.8, 0.2, 1] }}
          >
            {(title || eyebrow || description) && (
              <header
                data-testid={DATA_TEST_ID.HEADER}
                className="border-b border-line px-6 py-5"
              >
                {eyebrow && (
                  <div data-testid={DATA_TEST_ID.EYEBROW} className="eyebrow mb-1">
                    {eyebrow}
                  </div>
                )}
                {title && (
                  <h2
                    id={DATA_TEST_ID.TITLE}
                    data-testid={DATA_TEST_ID.TITLE}
                    className="h-page"
                    style={{ fontSize: 18 }}
                  >
                    {title}
                  </h2>
                )}
                {description && (
                  <p
                    data-testid={DATA_TEST_ID.DESCRIPTION}
                    className="mt-1 text-xs leading-relaxed text-muted"
                  >
                    {description}
                  </p>
                )}
              </header>
            )}
            <div
              data-testid={DATA_TEST_ID.BODY}
              className="flex-1 overflow-y-auto px-6 py-5"
            >
              {children}
            </div>
            {footer && (
              <footer
                data-testid={DATA_TEST_ID.FOOTER}
                className="border-t border-line bg-cream px-6 py-4"
              >
                {footer}
              </footer>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>,
    document.body,
  );
}
