/**
 * React hook that exposes the current Firebase auth state and the operations
 * the UI needs (sign-in, sign-up, sign-out). It deliberately does not surface
 * Firebase's User type — feature code receives a small, stable view.
 *
 * The hook also honors an optional React context override. In production
 * nothing mounts the override provider and the hook talks to live Firebase.
 * Storybook (and exploratory tooling) can wrap a subtree in
 * `<AuthOverrideProvider value={…} />` to inject a deterministic auth state.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import {
  createUserWithEmailAndPassword,
  signInWithEmailAndPassword,
  signOut as firebaseSignOut,
  onAuthStateChanged,
  type User as FirebaseUser,
} from "firebase/auth";
import { auth } from "../firebase";

/** Stable view of the signed-in user surfaced to feature code. */
export interface AuthenticatedUser {
  /** Firebase UID. */
  id: string;
  /** Account email. */
  email: string;
  /** Two-letter initials for the avatar (derived from email or display name). */
  initials: string;
}

export interface UseFirebaseAuthValue {
  /** The signed-in user, or null when signed out. */
  user: AuthenticatedUser | null;
  /** True while Firebase is resolving the initial auth state. */
  loading: boolean;
  signInWithEmail: (email: string, password: string) => Promise<void>;
  signUpWithEmail: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
}

function deriveInitialsFromUser(firebaseUser: FirebaseUser): string {
  const source = firebaseUser.displayName ?? firebaseUser.email ?? "";
  const cleaned = source.replace(/[^A-Za-z\s]/g, " ").trim();
  if (!cleaned) return "··";
  const words = cleaned.split(/\s+/);
  if (words.length >= 2) {
    return `${words[0][0]}${words[1][0]}`.toUpperCase();
  }
  return cleaned.slice(0, 2).toUpperCase();
}

function adaptFirebaseUser(firebaseUser: FirebaseUser): AuthenticatedUser {
  return {
    id: firebaseUser.uid,
    email: firebaseUser.email ?? "",
    initials: deriveInitialsFromUser(firebaseUser),
  };
}

// ── Override plumbing (Storybook / exploratory tooling only) ──────────────

const AuthOverrideContext = createContext<UseFirebaseAuthValue | null>(null);

export interface AuthOverrideProviderProps {
  value: UseFirebaseAuthValue;
  children: ReactNode;
}

/**
 * Wrap a subtree to short-circuit `useFirebaseAuth` and return a deterministic
 * value. Intended for Storybook and one-off exploratory harnesses — production
 * never mounts this.
 */
export function AuthOverrideProvider({
  value,
  children,
}: AuthOverrideProviderProps) {
  return (
    <AuthOverrideContext.Provider value={value}>
      {children}
    </AuthOverrideContext.Provider>
  );
}

// ── Hook ──────────────────────────────────────────────────────────────────

function useLiveFirebaseAuth(): UseFirebaseAuthValue {
  const [user, setUser] = useState<AuthenticatedUser | null>(() =>
    auth.currentUser ? adaptFirebaseUser(auth.currentUser) : null,
  );
  const [loading, setLoading] = useState<boolean>(auth.currentUser === null);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (firebaseUser) => {
      setUser(firebaseUser ? adaptFirebaseUser(firebaseUser) : null);
      setLoading(false);
    });
    return unsubscribe;
  }, []);

  const signInWithEmail = useCallback(async (email: string, password: string) => {
    await signInWithEmailAndPassword(auth, email, password);
  }, []);

  const signUpWithEmail = useCallback(async (email: string, password: string) => {
    await createUserWithEmailAndPassword(auth, email, password);
  }, []);

  const signOut = useCallback(async () => {
    await firebaseSignOut(auth);
  }, []);

  return { user, loading, signInWithEmail, signUpWithEmail, signOut };
}

export function useFirebaseAuth(): UseFirebaseAuthValue {
  // Stable rules-of-hooks order: both hooks are always called; the context
  // value just shadows the live result when present.
  const overrideValue = useContext(AuthOverrideContext);
  const liveValue = useLiveFirebaseAuth();
  return overrideValue ?? liveValue;
}
