import { cva } from "class-variance-authority";

export const buttonVariants = cva(
  // base: monospace label, rounded-md corners, smooth transitions, focus ring
  "inline-flex items-center justify-center gap-2 font-mono font-medium tracking-tight " +
    "border transition-colors duration-100 outline-none " +
    "focus-visible:ring-2 focus-visible:ring-navy/30 focus-visible:ring-offset-1 focus-visible:ring-offset-cream " +
    "disabled:opacity-45 disabled:cursor-not-allowed",
  {
    variants: {
      variant: {
        default:
          "bg-paper text-navy border-line-strong hover:bg-cream-200 hover:border-navy",
        primary:
          "bg-navy text-lime border-navy hover:bg-navy-700",
        lime:
          "bg-lime text-navy border-lime font-semibold hover:bg-lime-600 hover:border-lime-600",
        ghost:
          "bg-transparent text-navy border-line hover:bg-paper",
        danger:
          "bg-paper text-error border-line hover:bg-[#fcecea] hover:border-error",
      },
      size: {
        sm: "px-2.5 py-1 text-[11px] rounded",
        md: "px-3.5 py-2 text-xs rounded",
        lg: "px-4 py-2.5 text-[13px] rounded",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "md",
    },
  },
);
