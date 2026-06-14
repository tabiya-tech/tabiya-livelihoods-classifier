import { useEffect } from "react";
import type { Preview, StoryFn } from "@storybook/react";
import { I18nextProvider } from "react-i18next";
import { initialize, mswLoader } from "msw-storybook-addon";
import i18n from "../src/i18n/i18n";
import { Locale, LocalesLabels } from "../src/i18n/constants";
import "../src/index.css";

initialize({ onUnhandledRequest: "bypass" });

const localeToolbarItems = Object.entries(LocalesLabels).map(
  ([localeValue, label]) => ({ value: localeValue, title: label }),
);

const preview: Preview = {
  loaders: [mswLoader],
  parameters: {
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
  },
  globalTypes: {
    locale: {
      name: "Locale",
      description: "Internationalization locale",
      toolbar: {
        icon: "globe",
        items: localeToolbarItems,
        defaultValue: Locale.EN_US,
        showName: true,
      },
    },
  },
  decorators: [
    function StoryWithLocale(
      Story: StoryFn,
      context: { globals: { locale?: string } },
    ) {
      const selectedLocale = context.globals.locale ?? Locale.EN_US;

      useEffect(() => {
        i18n.changeLanguage(selectedLocale);
      }, [selectedLocale]);

      return (
        <I18nextProvider i18n={i18n}>
          <Story />
        </I18nextProvider>
      );
    },
  ],
};

export default preview;
