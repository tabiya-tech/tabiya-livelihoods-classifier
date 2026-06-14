import type { Meta, StoryObj } from "@storybook/react";
import { useTranslation } from "react-i18next";
import { userEvent, within } from "@storybook/test";
import { LanguageMenu, DATA_TEST_ID } from "./LanguageMenu";
import { LocalesLabels, SupportedLocales } from "@/i18n/constants";

const meta: Meta<typeof LanguageMenu> = {
  title: "i18n/LanguageMenu",
  component: LanguageMenu,
  parameters: { layout: "padded" },
};
export default meta;

type Story = StoryObj<typeof LanguageMenu>;

/**
 * Closed state. The trigger is the only visible affordance — clicking it
 * reveals the locale options. The accompanying `LiveTextEcho` re-renders on
 * every locale change so picking a language gives instant visible feedback.
 *
 * Note: the Storybook toolbar (top of the screen) also drives `i18n.language`
 * globally, so you can switch from either place.
 */
export const Default: Story = {
  render: () => (
    <div className="flex max-w-md flex-col gap-6">
      <div className="flex justify-end">
        <LanguageMenu />
      </div>
      <LiveTextEcho />
    </div>
  ),
};

/**
 * The menu rendered with the dropdown forced open via a `play` interaction.
 * Useful for visually reviewing option styling without having to click
 * through every story.
 */
export const Open: Story = {
  render: () => (
    <div className="flex justify-end pb-32">
      <LanguageMenu />
    </div>
  ),
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const triggerButton = await canvas.findByTestId(DATA_TEST_ID.TRIGGER);
    await userEvent.click(triggerButton);
  },
};

/**
 * The menu in its real Topbar context: a navy surface with the trigger
 * anchored to the right edge. Confirms the dropdown doesn't pick up the
 * dark background and the trigger icon stays legible.
 */
export const InTopbarContext: Story = {
  render: () => (
    <div className="flex items-center justify-between rounded-md border border-line bg-cream px-6 py-3">
      <span className="font-mono text-xs text-muted">Workspace / Dashboard</span>
      <LanguageMenu />
    </div>
  ),
};

/**
 * Every label rendered side-by-side so translators can spot label width
 * issues at a glance.
 */
export const AllLocaleLabels: Story = {
  render: () => (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-muted">
          <th className="py-2 pr-6 font-mono">Locale</th>
          <th className="py-2 font-mono">Label</th>
        </tr>
      </thead>
      <tbody>
        {SupportedLocales.map((supportedLocale) => (
          <tr key={supportedLocale} className="border-t border-line">
            <td className="py-2 pr-6 font-mono text-xs text-navy">
              {supportedLocale}
            </td>
            <td className="py-2">{LocalesLabels[supportedLocale]}</td>
          </tr>
        ))}
      </tbody>
    </table>
  ),
};

function LiveTextEcho() {
  const { t, i18n } = useTranslation();
  return (
    <div className="space-y-2 rounded-md border border-line bg-paper p-4 font-mono text-sm">
      <div className="text-xs uppercase tracking-wider text-muted">
        Live i18n echo
      </div>
      <div>
        <span className="text-muted">i18n.language:</span>{" "}
        <span className="text-navy">{i18n.language}</span>
      </div>
      <div>
        <span className="text-muted">common.buttons.signIn:</span>{" "}
        <span className="text-navy">{t("common.buttons.signIn")}</span>
      </div>
      <div>
        <span className="text-muted">shell.topbar.apiHealthy:</span>{" "}
        <span className="text-navy">{t("shell.topbar.apiHealthy")}</span>
      </div>
      <div>
        <span className="text-muted">auth.login.signInHeading:</span>{" "}
        <span className="text-navy">{t("auth.login.signInHeading")}</span>
      </div>
    </div>
  );
}
