/**
 * Split-screen Firebase sign-in / sign-up.
 *
 * Layout matches the design handoff: a navy column on the left with the brand
 * mark, an editorial tagline, and a version footer; a cream paper card on the
 * right with the email/password form. The toggle at the bottom of the card
 * swaps between sign-in and sign-up.
 *
 * Bouncing to /dashboard when the user is already signed in is handled by the
 * PublicOnlyRoute wrapper, not by this component. Once Firebase reports an
 * authenticated user, that wrapper unmounts LoginPage and navigates away.
 */

import { useState, type FormEvent } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Button, FormField, Icon, Input } from "@/components";
import { useFirebaseAuth } from "@/lib/auth/useFirebaseAuth";

const uniqueId = "1c6f0d6d-fb1c-4f7e-a6ed-6b9a9a3eebbd";

export const DATA_TEST_ID = {
  CONTAINER: `login-page-container-${uniqueId}`,
  LEFT_PANEL: `login-page-left-panel-${uniqueId}`,
  RIGHT_PANEL: `login-page-right-panel-${uniqueId}`,
  HEADING: `login-page-heading-${uniqueId}`,
  EMAIL_INPUT: `login-page-email-input-${uniqueId}`,
  PASSWORD_INPUT: `login-page-password-input-${uniqueId}`,
  SUBMIT_BUTTON: `login-page-submit-button-${uniqueId}`,
  TOGGLE_BUTTON: `login-page-toggle-button-${uniqueId}`,
  ERROR_MESSAGE: `login-page-error-message-${uniqueId}`,
};

export function LoginPage() {
  const { t } = useTranslation();
  const { signInWithEmail, signUpWithEmail } = useFirebaseAuth();

  const [mode, setMode] = useState<"sign-in" | "sign-up">("sign-in");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setErrorMessage(null);
    try {
      if (mode === "sign-in") {
        await signInWithEmail(email, password);
      } else {
        await signUpWithEmail(email, password);
      }
    } catch (caught) {
      setErrorMessage(
        caught instanceof Error ? caught.message : t("auth.login.errorFallback"),
      );
    } finally {
      setSubmitting(false);
    }
  }

  const isSignUp = mode === "sign-up";
  const headingText = isSignUp
    ? t("auth.login.signUpHeading")
    : t("auth.login.signInHeading");
  const subheadingText = isSignUp
    ? t("auth.login.signUpSubheading")
    : t("auth.login.signInSubheading");
  const submitLabel = isSignUp
    ? t("common.buttons.createAccount")
    : t("common.buttons.signIn");
  const toggleLabel = isSignUp
    ? t("auth.login.toggleToSignIn")
    : t("auth.login.toggleToSignUp");

  return (
    <div
      data-testid={DATA_TEST_ID.CONTAINER}
      className="grid min-h-screen bg-cream md:grid-cols-[1.1fr_1fr]"
    >
      <aside
        data-testid={DATA_TEST_ID.LEFT_PANEL}
        className="relative flex flex-col bg-navy px-14 py-12 text-cream"
      >
        <div className="mb-auto flex items-center gap-3">
          <div className="grid h-7 w-7 place-items-center rounded bg-lime font-mono text-sm font-bold text-navy">
            T
          </div>
          <div className="font-mono text-base leading-tight">
            {t("shell.brand.name")}
            <span className="mt-0.5 block text-[11px] font-normal tracking-wide text-cream/55">
              {t("shell.brand.product")}
            </span>
          </div>
        </div>

        <div className="relative z-10 my-6 max-w-md">
          <div className="eyebrow text-cream/55">
            {t("auth.login.tagline.eyebrow")}
          </div>
          <p className="mt-3 font-serif text-2xl leading-snug text-cream">
            <Trans
              i18nKey="auth.login.tagline.body"
              components={{
                occupations: <span className="text-lime" />,
                skills: <span className="text-lime" />,
                qualifications: <span className="text-lime" />,
              }}
            />
          </p>
        </div>

        <div className="relative z-10 font-mono text-[11px] text-cream/50">
          {t("auth.login.versionFooter")}
        </div>

        <div
          aria-hidden
          className="pointer-events-none absolute inset-x-0 bottom-0 h-3/5 bg-[radial-gradient(circle_at_30%_80%,rgba(0,255,145,0.1),transparent_60%)]"
        />
      </aside>

      <main
        data-testid={DATA_TEST_ID.RIGHT_PANEL}
        className="grid place-items-center px-8 py-12"
      >
        <div className="w-full max-w-sm rounded-lg border border-line bg-paper p-9">
          <h1
            data-testid={DATA_TEST_ID.HEADING}
            className="h-page mb-1.5"
            style={{ fontSize: 22 }}
          >
            {headingText}
          </h1>
          <p className="mb-6 text-sm text-muted">{subheadingText}</p>

          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <FormField label={t("common.fields.email")} required>
              <Input
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                autoComplete={isSignUp ? "email" : "username"}
                placeholder={t("auth.login.emailPlaceholder")}
                required
                data-testid={DATA_TEST_ID.EMAIL_INPUT}
              />
            </FormField>

            <FormField label={t("common.fields.password")} required>
              <Input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete={isSignUp ? "new-password" : "current-password"}
                placeholder={t("auth.login.passwordPlaceholder")}
                required
                data-testid={DATA_TEST_ID.PASSWORD_INPUT}
              />
            </FormField>

            {errorMessage && (
              <p
                data-testid={DATA_TEST_ID.ERROR_MESSAGE}
                role="alert"
                className="text-xs leading-snug text-error"
              >
                {errorMessage}
              </p>
            )}

            <Button
              type="submit"
              variant="primary"
              size="lg"
              className="w-full justify-center"
              loading={submitting}
              trailing={!submitting && <Icon name="arrowRight" />}
              data-testid={DATA_TEST_ID.SUBMIT_BUTTON}
            >
              {submitLabel}
            </Button>
          </form>

          <div className="mt-5 text-center">
            <button
              type="button"
              data-testid={DATA_TEST_ID.TOGGLE_BUTTON}
              onClick={() => {
                setMode(isSignUp ? "sign-in" : "sign-up");
                setErrorMessage(null);
              }}
              className="font-mono text-xs text-muted hover:text-navy"
            >
              {toggleLabel}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
