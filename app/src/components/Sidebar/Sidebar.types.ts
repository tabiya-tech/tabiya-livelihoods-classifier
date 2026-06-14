import type { IconName } from "@/components";

export interface SidebarNavItem {
  id: string;
  label: string;
  icon: IconName;
}

export interface SidebarNavGroup {
  label: string;
  items: SidebarNavItem[];
}

export interface SidebarUser {
  /** Two-letter initials shown in the avatar circle. */
  initials: string;
  /** Display label (email or name). */
  label: string;
}
