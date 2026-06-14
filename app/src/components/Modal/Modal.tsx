import { useEffect, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";
import { IconButton } from "@/components";

const uniqueId = "03331cdf-f1a1-49e9-8114-6055ce43a17c";

export const DATA_TEST_ID = {
  BACKDROP: `modal-backdrop-${uniqueId}`,
  DIALOG: `modal-dialog-${uniqueId}`,
  TITLE: `modal-title-${uniqueId}`,
  DESCRIPTION: `modal-description-${uniqueId}`,
  BODY: `modal-body-${uniqueId}`,
  FOOTER: `modal-footer-${uniqueId}`,
  CLOSE_BUTTON: `modal-close-button-${uniqueId}`,
};

export interface ModalProps {
  open: boolean;
  onClose: () => void;
  /** Accessible title — visible heading + aria-labelledby. */
  title?: ReactNode;
  /** Optional supporting description below the title. */
  description?: ReactNode;
  /** Footer slot — typically Cancel/Confirm buttons. */
  footer?: ReactNode;
  /** Custom width; defaults to 480px. */
  width?: number | string;
  className?: string;
  children?: ReactNode;
}

const backdropVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1 },
};

const dialogVariants = {
  hidden: { opacity: 0, scale: 0.96, y: 8 },
  visible: { opacity: 1, scale: 1, y: 0 },
};

export function Modal({
  open,
  onClose,
  title,
  description,
  footer,
  width = 480,
  className,
  children,
}: ModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    dialogRef.current?.focus();
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  return createPortal(
    <AnimatePresence>
      {open && (
        <motion.div
          role="presentation"
          data-testid={DATA_TEST_ID.BACKDROP}
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) onClose();
          }}
          className="fixed inset-0 z-[200] flex items-center justify-center bg-ink/35 p-4"
          variants={backdropVariants}
          initial="hidden"
          animate="visible"
          exit="hidden"
          transition={{ duration: 0.16, ease: "easeOut" }}
        >
          <motion.div
            ref={dialogRef}
            role="dialog"
            data-testid={DATA_TEST_ID.DIALOG}
            aria-modal="true"
            aria-labelledby={title ? DATA_TEST_ID.TITLE : undefined}
            tabIndex={-1}
            style={{ width }}
            className={mergeClassNames(
              "max-h-[90vh] overflow-y-auto rounded-lg border border-line bg-paper p-6 shadow-card-2 outline-none",
              className,
            )}
            variants={dialogVariants}
            initial="hidden"
            animate="visible"
            exit="hidden"
            transition={{ duration: 0.18, ease: [0.2, 0.8, 0.2, 1] }}
          >
            <header className="mb-4 flex items-start justify-between gap-4">
              <div>
                {title && (
                  <h2
                    id={DATA_TEST_ID.TITLE}
                    data-testid={DATA_TEST_ID.TITLE}
                    className="m-0 font-mono text-base font-medium text-navy"
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
              </div>
              <IconButton
                icon="close"
                aria-label="Close"
                onClick={onClose}
                size="sm"
                data-testid={DATA_TEST_ID.CLOSE_BUTTON}
              />
            </header>
            <div data-testid={DATA_TEST_ID.BODY}>{children}</div>
            {footer && (
              <footer
                data-testid={DATA_TEST_ID.FOOTER}
                className="mt-6 flex items-center justify-end gap-2"
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
