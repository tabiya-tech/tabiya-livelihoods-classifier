import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import i18n from "@/i18n/i18n";

const mockSignInWithEmail = vi.fn();
const mockSignUpWithEmail = vi.fn();
const mockUseFirebaseAuth = vi.fn();
vi.mock("@/lib/auth/useFirebaseAuth", () => ({
  useFirebaseAuth: () => mockUseFirebaseAuth(),
}));

import { LoginPage, DATA_TEST_ID } from "./LoginPage";

beforeEach(() => {
  mockSignInWithEmail.mockReset();
  mockSignUpWithEmail.mockReset();
  mockUseFirebaseAuth.mockReturnValue({
    user: null,
    loading: false,
    signInWithEmail: mockSignInWithEmail,
    signUpWithEmail: mockSignUpWithEmail,
    signOut: vi.fn(),
  });
});

describe("LoginPage", () => {
  it("renders the sign-in heading by default", () => {
    // GIVEN the LoginPage in its default state and the expected heading text from i18n
    const expectedHeadingText = i18n.t("auth.login.signInHeading");

    // WHEN we render it
    render(<LoginPage />);

    // THEN the heading shows sign-in copy and the email/password inputs are present
    expect(screen.getByTestId(DATA_TEST_ID.HEADING)).toHaveTextContent(
      expectedHeadingText,
    );
    expect(screen.getByTestId(DATA_TEST_ID.EMAIL_INPUT)).toBeInTheDocument();
    expect(screen.getByTestId(DATA_TEST_ID.PASSWORD_INPUT)).toBeInTheDocument();
  });

  it("calls signInWithEmail with the typed credentials on submit", async () => {
    // GIVEN the credentials the user is going to enter
    const givenEmail = "sara.m@tabiya.org";
    const givenPassword = "hunter2!";
    mockSignInWithEmail.mockResolvedValue(undefined);
    render(<LoginPage />);

    // WHEN the user types into both inputs and submits
    await userEvent.type(screen.getByTestId(DATA_TEST_ID.EMAIL_INPUT), givenEmail);
    await userEvent.type(
      screen.getByTestId(DATA_TEST_ID.PASSWORD_INPUT),
      givenPassword,
    );
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.SUBMIT_BUTTON));

    // THEN signInWithEmail is invoked with the given credentials
    expect(mockSignInWithEmail).toHaveBeenCalledWith(givenEmail, givenPassword);
  });

  it("toggles to sign-up mode and uses signUpWithEmail on submit", async () => {
    // GIVEN the credentials the new user will enter and the expected sign-up heading
    const givenEmail = "new.user@tabiya.org";
    const givenPassword = "fresh-account-1";
    const expectedSignUpHeadingText = i18n.t("auth.login.signUpHeading");
    mockSignUpWithEmail.mockResolvedValue(undefined);
    render(<LoginPage />);

    // WHEN the user toggles to sign-up
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.TOGGLE_BUTTON));

    // AND submits the form
    await userEvent.type(screen.getByTestId(DATA_TEST_ID.EMAIL_INPUT), givenEmail);
    await userEvent.type(
      screen.getByTestId(DATA_TEST_ID.PASSWORD_INPUT),
      givenPassword,
    );
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.SUBMIT_BUTTON));

    // THEN the heading reflects sign-up and signUpWithEmail is called
    expect(screen.getByTestId(DATA_TEST_ID.HEADING)).toHaveTextContent(
      expectedSignUpHeadingText,
    );
    expect(mockSignUpWithEmail).toHaveBeenCalledWith(givenEmail, givenPassword);
    expect(mockSignInWithEmail).not.toHaveBeenCalled();
  });

  it("surfaces the Firebase error message when sign-in rejects", async () => {
    // GIVEN sign-in throws with a meaningful message
    const givenErrorMessage = "auth/invalid-credential";
    mockSignInWithEmail.mockRejectedValue(new Error(givenErrorMessage));
    render(<LoginPage />);

    // WHEN the user submits the form
    await userEvent.type(
      screen.getByTestId(DATA_TEST_ID.EMAIL_INPUT),
      "x@tabiya.org",
    );
    await userEvent.type(screen.getByTestId(DATA_TEST_ID.PASSWORD_INPUT), "bad");
    await userEvent.click(screen.getByTestId(DATA_TEST_ID.SUBMIT_BUTTON));

    // THEN the inline error renders with the given message
    await waitFor(() =>
      expect(screen.getByTestId(DATA_TEST_ID.ERROR_MESSAGE)).toHaveTextContent(
        givenErrorMessage,
      ),
    );
  });
});
