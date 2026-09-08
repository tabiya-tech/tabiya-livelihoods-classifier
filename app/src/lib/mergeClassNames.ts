import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Compose Tailwind class names. clsx handles conditionals/arrays;
 * tailwind-merge resolves conflicts (e.g. `px-2 px-4` → `px-4`).
 */
export function mergeClassNames(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
