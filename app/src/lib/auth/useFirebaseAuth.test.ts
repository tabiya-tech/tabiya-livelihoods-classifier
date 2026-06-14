import { describe, expect, it, vi, beforeEach } from "vitest";
import { act, renderHook } from "@testing-library/react";

// vi.hoisted lets us share refs with vi.mock factories without TDZ errors.
const firebaseAuthMocks = vi.hoisted(() => {
  return {
    onAuthStateChanged: vi.fn(),
    signInWithEmailAndPassword: vi.fn(),
    createUserWithEmailAndPassword: vi.fn(),
    signOut: vi.fn(),
    fakeAuthInstance: {
      currentUser: null as
        | null
        | { uid: string; email: string; displayName: string | null },
    },
  };
});

vi.mock("firebase/auth", () => ({
  onAuthStateChanged: firebaseAuthMocks.onAuthStateChanged,
  signInWithEmailAndPassword: firebaseAuthMocks.signInWithEmailAndPassword,
  createUserWithEmailAndPassword: firebaseAuthMocks.createUserWithEmailAndPassword,
  signOut: firebaseAuthMocks.signOut,
  getAuth: () => firebaseAuthMocks.fakeAuthInstance,
}));

vi.mock("firebase/app", () => ({
  initializeApp: () => ({}),
}));

// Also stub `@/lib/firebase` directly because the real module may have been
// cached by the test setup (which imports MSW handlers that transitively
// load it). With the real cached `auth` constant in play, this hook's
// `signIn`/`signOut` callbacks would receive the live Firebase auth object
// instead of our fake. The mock below replaces the module-level constant
// in `@/lib/firebase` with the fake auth instance our other mocks use.
vi.mock("@/lib/firebase", () => ({
  firebaseApp: {},
  auth: firebaseAuthMocks.fakeAuthInstance,
}));

// Import after mocks so the module under test picks them up.
import { useFirebaseAuth } from "./useFirebaseAuth";

beforeEach(() => {
  firebaseAuthMocks.onAuthStateChanged.mockReset();
  firebaseAuthMocks.signInWithEmailAndPassword.mockReset();
  firebaseAuthMocks.createUserWithEmailAndPassword.mockReset();
  firebaseAuthMocks.signOut.mockReset();
  firebaseAuthMocks.fakeAuthInstance.currentUser = null;
});

describe("useFirebaseAuth", () => {
  it("starts in loading state until Firebase resolves the auth state", () => {
    // GIVEN onAuthStateChanged that never fires synchronously
    firebaseAuthMocks.onAuthStateChanged.mockImplementation(() => () => {});

    // WHEN we mount the hook
    const { result } = renderHook(() => useFirebaseAuth());

    // THEN it reports loading and no user
    expect(result.current.loading).toBe(true);
    expect(result.current.user).toBeNull();
  });

  it("exposes the signed-in user when Firebase emits a non-null user", () => {
    // GIVEN onAuthStateChanged that fires synchronously with a user
    const givenFirebaseUser = {
      uid: "user-uid-1",
      email: "sara.m@tabiya.org",
      displayName: null,
    };
    firebaseAuthMocks.onAuthStateChanged.mockImplementation((_auth, callback) => {
      callback(givenFirebaseUser);
      return () => {};
    });

    // WHEN we mount the hook
    const { result } = renderHook(() => useFirebaseAuth());

    // THEN it reports the adapted user and is no longer loading
    expect(result.current.loading).toBe(false);
    expect(result.current.user).toEqual({
      id: "user-uid-1",
      email: "sara.m@tabiya.org",
      // sara m tabiya org → first letters of the first two non-empty words
      initials: "SM",
    });
  });

  it("clears the user when Firebase emits null (sign-out)", () => {
    // GIVEN an emitter we can drive imperatively
    let emitAuthState: (user: unknown) => void = () => {};
    firebaseAuthMocks.onAuthStateChanged.mockImplementation((_auth, callback) => {
      emitAuthState = callback;
      return () => {};
    });
    const { result } = renderHook(() => useFirebaseAuth());

    // AND an initial signed-in user
    act(() =>
      emitAuthState({
        uid: "uid",
        email: "alex@tabiya.org",
        displayName: null,
      }),
    );
    expect(result.current.user?.email).toBe("alex@tabiya.org");

    // WHEN Firebase emits null (the user signed out)
    act(() => emitAuthState(null));

    // THEN the hook reports no user
    expect(result.current.user).toBeNull();
  });

  it("invokes Firebase signInWithEmailAndPassword with the given credentials", async () => {
    // GIVEN a credential pair
    const givenEmail = "alex@tabiya.org";
    const givenPassword = "hunter2";
    firebaseAuthMocks.onAuthStateChanged.mockImplementation(() => () => {});
    firebaseAuthMocks.signInWithEmailAndPassword.mockResolvedValue({});
    const { result } = renderHook(() => useFirebaseAuth());

    // WHEN we call signInWithEmail
    await act(async () => {
      await result.current.signInWithEmail(givenEmail, givenPassword);
    });

    // THEN Firebase's signInWithEmailAndPassword is called with those credentials
    expect(firebaseAuthMocks.signInWithEmailAndPassword).toHaveBeenCalledWith(
      firebaseAuthMocks.fakeAuthInstance,
      givenEmail,
      givenPassword,
    );
  });

  it("invokes Firebase signOut", async () => {
    // GIVEN a mounted hook
    firebaseAuthMocks.onAuthStateChanged.mockImplementation(() => () => {});
    firebaseAuthMocks.signOut.mockResolvedValue(undefined);
    const { result } = renderHook(() => useFirebaseAuth());

    // WHEN we call signOut
    await act(async () => {
      await result.current.signOut();
    });

    // THEN Firebase's signOut is called
    expect(firebaseAuthMocks.signOut).toHaveBeenCalledWith(
      firebaseAuthMocks.fakeAuthInstance,
    );
  });
});
