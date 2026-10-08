import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { AnimatePresence, motion } from "framer-motion";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";
import type {
  ToastContextValue,
  ToastInput,
  ToastItem,
  ToastPlacement,
  ToastTone,
} from "./Toast.types";

const uniqueId = "c6105162-3fdc-48f0-a569-098942aa97f2";

export const DATA_TEST_ID = {
  VIEWPORT: `toast-viewport-${uniqueId}`,
  ITEM: `toast-item-${uniqueId}`,
  ITEM_DOT: `toast-item-dot-${uniqueId}`,
};

const toneClass: Record<ToastTone, string> = {
  info: "bg-line-strong",
  success: "bg-lime-600",
  error: "bg-error",
};

const placementClass: Record<ToastPlacement, string> = {
  "top-left": "top-6 left-6 items-start",
  "top-center": "top-6 left-1/2 -translate-x-1/2 items-center",
  "top-right": "top-6 right-6 items-end",
  "bottom-left": "bottom-6 left-6 items-start",
  "bottom-center": "bottom-6 left-1/2 -translate-x-1/2 items-center",
  "bottom-right": "bottom-6 right-6 items-end",
};

function getEntryOffset(placement: ToastPlacement): { x: number; y: number } {
  if (placement.startsWith("top")) return { x: 0, y: -12 };
  return { x: 0, y: 12 };
}

let counter = 0;
function nextId() {
  counter += 1;
  return `toast-${counter}`;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export interface ToastProviderProps {
  children: ReactNode;
  /** Where on the viewport the toast stack renders. Defaults to 'bottom-right'. */
  placement?: ToastPlacement;
}

export function ToastProvider({
  children,
  placement = "bottom-right",
}: ToastProviderProps) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const timersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const dismiss = useCallback((id: string) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
    const timer = timersRef.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timersRef.current.delete(id);
    }
  }, []);

  const show = useCallback(
    (input: ToastInput) => {
      const id = nextId();
      const duration = input.durationMs ?? 2400;
      setToasts((current) => [...current, { ...input, id }]);
      if (duration > 0) {
        const timer = setTimeout(() => dismiss(id), duration);
        timersRef.current.set(id, timer);
      }
      return id;
    },
    [dismiss],
  );

  const value = useMemo(() => ({ show, dismiss }), [show, dismiss]);
  const entry = getEntryOffset(placement);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div
        aria-live="polite"
        aria-atomic="false"
        data-testid={DATA_TEST_ID.VIEWPORT}
        data-placement={placement}
        className={mergeClassNames(
          "pointer-events-none fixed z-[300] flex flex-col gap-2",
          placementClass[placement],
        )}
      >
        <AnimatePresence initial={false}>
          {toasts.map((toast) => (
            <motion.div
              key={toast.id}
              role="status"
              data-testid={DATA_TEST_ID.ITEM}
              initial={{ opacity: 0, ...entry, scale: 0.96 }}
              animate={{ opacity: 1, x: 0, y: 0, scale: 1 }}
              exit={{ opacity: 0, ...entry, scale: 0.96 }}
              transition={{ duration: 0.18, ease: [0.2, 0.8, 0.2, 1] }}
              className={mergeClassNames(
                "pointer-events-auto flex items-center gap-2 rounded-md border border-line",
                "bg-paper px-3.5 py-2 font-mono text-xs text-navy shadow-card-2",
              )}
            >
              <span
                aria-hidden
                data-testid={DATA_TEST_ID.ITEM_DOT}
                className={mergeClassNames(
                  "h-1.5 w-1.5 rounded-full",
                  toneClass[toast.tone ?? "info"],
                )}
              />
              {toast.message}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error("useToast must be used inside a ToastProvider");
  }
  return context;
}
