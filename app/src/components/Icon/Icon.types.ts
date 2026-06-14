import type { SVGProps } from "react";

export type IconName =
  | "classify"
  | "dashboard"
  | "config"
  | "key"
  | "docs"
  | "history"
  | "copy"
  | "arrowRight"
  | "external"
  | "plus"
  | "trash"
  | "check"
  | "close"
  | "filter"
  | "download"
  | "upload"
  | "spark"
  | "search";

export interface IconProps extends SVGProps<SVGSVGElement> {
  name: IconName;
  /** Pixel size for both width and height. Defaults to 14. */
  size?: number;
}
