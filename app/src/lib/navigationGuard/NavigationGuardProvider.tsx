/**
 * App-wide navigation guard.
 *
 * Pages that have unsaved work register a `confirmLeave` callback. Anything
 * that initiates a navigation (Sidebar, Topbar's brand click, sign-out flow)
 * routes intent through {@link useNavigationGuard}.requestNavigate, which
 * either invokes the page's confirmation or proceeds immediately when there
 * is no guard registered.
 *
 * The guard is intentionally lightweight — it doesn't intercept browser-level
 * navigations (back button, address bar). Pages that need that should also
 * register a `beforeunload` listener for the tab close case.
 */

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

export interface NavigationGuardContextValue {
  /**
   * Pages register their "Are you sure?" confirmation here. Returning a
   * boolean (or a promise of one) decides whether the navigation proceeds.
   * Pass `null` to clear the guard (e.g. on unmount / save success).
   */
  registerGuard: (confirmLeave: GuardConfirmFn | null) => void;
  /**
   * Initiate a navigation. If a guard is registered it's invoked first; the
   * navigation runs only when the guard resolves truthy.
   */
  requestNavigate: (proceed: () => void | Promise<void>) => Promise<void>;
  /** True when a guard is currently registered. */
  hasActiveGuard: boolean;
}

export type GuardConfirmFn = () => Promise<boolean> | boolean;

const NavigationGuardContext =
  createContext<NavigationGuardContextValue | null>(null);

export interface NavigationGuardProviderProps {
  children: ReactNode;
}

export function NavigationGuardProvider({
  children,
}: NavigationGuardProviderProps) {
  const guardRef = useRef<GuardConfirmFn | null>(null);
  const [hasActiveGuard, setHasActiveGuard] = useState(false);

  const registerGuard = useCallback((confirmLeave: GuardConfirmFn | null) => {
    guardRef.current = confirmLeave;
    setHasActiveGuard(confirmLeave !== null);
  }, []);

  const requestNavigate = useCallback(
    async (proceed: () => void | Promise<void>) => {
      const guard = guardRef.current;
      if (!guard) {
        await proceed();
        return;
      }
      const allow = await guard();
      if (allow) {
        await proceed();
      }
    },
    [],
  );

  const value = useMemo(
    () => ({ registerGuard, requestNavigate, hasActiveGuard }),
    [registerGuard, requestNavigate, hasActiveGuard],
  );

  return (
    <NavigationGuardContext.Provider value={value}>
      {children}
    </NavigationGuardContext.Provider>
  );
}

export function useNavigationGuard(): NavigationGuardContextValue {
  const context = useContext(NavigationGuardContext);
  if (!context) {
    throw new Error(
      "useNavigationGuard must be used inside a NavigationGuardProvider",
    );
  }
  return context;
}
