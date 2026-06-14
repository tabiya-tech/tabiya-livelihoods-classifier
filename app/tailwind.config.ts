import type { Config } from "tailwindcss";
import {
  borderRadius,
  boxShadow,
  colors,
  fontFamily,
} from "./src/theme/theme";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors,
      fontFamily,
      borderRadius,
      boxShadow,
      fontSize: {
        eyebrow: ["11px", { lineHeight: "1.4", letterSpacing: "0.12em" }],
      },
    },
  },
  plugins: [],
};

export default config;
