import { cva } from "class-variance-authority";

export const tagVariants = cva(
  "inline-flex items-center gap-1.5 font-mono whitespace-nowrap rounded-sm border",
  {
    variants: {
      tone: {
        neutral: "bg-cream-200 text-navy border-line",
        lime: "bg-lime text-navy border-lime",
        teal: "bg-paper text-teal border-line",
        muted: "bg-paper text-muted border-line",
        danger: "bg-paper text-error border-line",
      },
      size: {
        sm: "px-1.5 py-px text-[10px]",
        md: "px-2 py-0.5 text-[11px]",
      },
    },
    defaultVariants: { tone: "neutral", size: "md" },
  },
);
