/**
 * Wires a page's dirty state into the app-wide navigation guard and into
 * the browser's beforeunload event.
 *
 * Exposes:
 * - `isPromptOpen` — true while the user is being asked to confirm leaving.
 * - `confirm()` — caller resolves the prompt as "leave anyway".
 * - `cancel()` — caller resolves the prompt as "stay".
 *
 * The hook resolves a single in-flight prompt at a time. When `isDirty` is
 * false the guard unregisters itself and the prompt never opens.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigationGuard } from "@/lib/navigationGuard";

type PromptResolver = (allow: boolean) => void;

export interface UnsavedChangesGuardState {
  isPromptOpen: boolean;
  confirm: () => void;
  cancel: () => void;
}

export function useUnsavedChangesGuard(
  isDirty: boolean,
): UnsavedChangesGuardState {
  const { registerGuard } = useNavigationGuard();
  const [isPromptOpen, setIsPromptOpen] = useState(false);
  const resolverRef = useRef<PromptResolver | null>(null);

  useEffect(() => {
    if (!isDirty) {
      registerGuard(null);
      return;
    }
    registerGuard(
      () =>
        new Promise<boolean>((resolve) => {
          resolverRef.current = resolve;
          setIsPromptOpen(true);
        }),
    );
    return () => {
      registerGuard(null);
    };
  }, [isDirty, registerGuard]);

  useEffect(() => {
    if (!isDirty) return;
    function onBeforeUnload(event: BeforeUnloadEvent) {
      event.preventDefault();
      event.returnValue = "";
    }
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [isDirty]);

  const resolveAndClose = useCallback((allow: boolean) => {
    const resolver = resolverRef.current;
    resolverRef.current = null;
    setIsPromptOpen(false);
    if (resolver) resolver(allow);
  }, []);

  const confirm = useCallback(() => resolveAndClose(true), [resolveAndClose]);
  const cancel = useCallback(() => resolveAndClose(false), [resolveAndClose]);

  return { isPromptOpen, confirm, cancel };
}
