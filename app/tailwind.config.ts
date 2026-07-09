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
      keyframes: {
        shake: {
          "0%, 100%": { transform: "translateX(0)" },
          "20%": { transform: "translateX(-3px)" },
          "40%": { transform: "translateX(3px)" },
          "60%": { transform: "translateX(-2px)" },
          "80%": { transform: "translateX(2px)" },
        },
      },
      animation: {
        shake: "shake 0.3s ease-in-out",
      },
    },
  },
  plugins: [],
};

export default config;
