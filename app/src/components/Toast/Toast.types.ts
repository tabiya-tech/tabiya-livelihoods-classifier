import type { ReactNode } from "react";

export type ToastTone = "info" | "success" | "error";

export type ToastPlacement =
  | "top-left"
  | "top-center"
  | "top-right"
  | "bottom-left"
  | "bottom-center"
  | "bottom-right";

export interface ToastInput {
  /** Visible label. */
  message: ReactNode;
  /** Tone — drives the leading dot color. Defaults to 'info'. */
  tone?: ToastTone;
  /** Auto-dismiss duration in ms. Defaults to 2400. Set to 0 to disable. */
  durationMs?: number;
}

export interface ToastItem extends ToastInput {
  id: string;
}

export interface ToastContextValue {
  show: (input: ToastInput) => string;
  dismiss: (id: string) => void;
}
