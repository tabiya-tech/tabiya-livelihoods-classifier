import { Fragment } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";
import { Icon } from "@/components";
import { NavLink } from "@/components";
import type { SidebarNavGroup, SidebarUser } from "./Sidebar.types";

const uniqueId = "152d6364-a850-4cc4-bc37-e9e58d5965f5";

export const DATA_TEST_ID = {
  CONTAINER: `sidebar-container-${uniqueId}`,
  BRAND_BUTTON: `sidebar-brand-button-${uniqueId}`,
  GROUP_LABEL: `sidebar-group-label-${uniqueId}`,
  USER_INFO: `sidebar-user-info-${uniqueId}`,
  USER_AVATAR: `sidebar-user-avatar-${uniqueId}`,
  SIGN_OUT: `sidebar-sign-out-${uniqueId}`,
};

export interface SidebarProps {
  /** Currently active route id (matches a SidebarNavItem.id). */
  activeId: string;
  /** Called when the user clicks a nav item. */
  onNavigate: (id: string) => void;
  /** Grouped navigation links. */
  groups: SidebarNavGroup[];
  /** Brand mark click handler — typically navigates to the home route. */
  onBrandClick?: () => void;
  /** Signed-in user, shown in the footer. */
  user?: SidebarUser;
  /** Sign-out handler — when omitted, the button is hidden. */
  onSignOut?: () => void;
  className?: string;
}

export function Sidebar({
  activeId,
  onNavigate,
  groups,
  onBrandClick,
  user,
  onSignOut,
  className,
}: SidebarProps) {
  return (
    <aside
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        "sticky top-0 flex h-screen w-[232px] flex-col gap-1 bg-navy px-4 py-7 text-cream",
        className,
      )}
    >
      <button
        type="button"
        data-testid={DATA_TEST_ID.BRAND_BUTTON}
        onClick={onBrandClick}
        className="mb-6 flex items-center gap-2.5 bg-transparent p-1.5 text-left text-inherit"
      >
        <div className="grid h-7 w-7 place-items-center rounded bg-lime font-mono text-sm font-bold text-navy">
          T
        </div>
        <div className="font-mono text-sm leading-tight">
          Tabiya
          <span className="mt-0.5 block text-[11px] font-normal tracking-wide text-cream/55">
            Classifier
          </span>
        </div>
      </button>

      {groups.map((group) => (
        <Fragment key={group.label}>
          <div
            data-testid={DATA_TEST_ID.GROUP_LABEL}
            className="px-2.5 pb-1.5 pt-3.5 font-mono text-[10px] uppercase tracking-[0.14em] text-cream/40"
          >
            {group.label}
          </div>
          {group.items.map((item) => (
            <NavLink
              key={item.id}
              active={item.id === activeId}
              icon={<Icon name={item.icon} />}
              onClick={() => onNavigate(item.id)}
            >
              {item.label}
            </NavLink>
          ))}
        </Fragment>
      ))}

      {(user || onSignOut) && (
        <div className="mt-auto border-t border-cream/10 pt-4">
          {user && (
            <div
              data-testid={DATA_TEST_ID.USER_INFO}
              className="flex items-center gap-2.5 px-2 py-1.5 text-xs text-cream/70"
            >
              <div
                data-testid={DATA_TEST_ID.USER_AVATAR}
                className="grid h-6 w-6 place-items-center rounded-full bg-cream font-mono text-[11px] font-semibold text-navy"
              >
                {user.initials}
              </div>
              <div className="overflow-hidden text-ellipsis whitespace-nowrap">
                {user.label}
              </div>
            </div>
          )}
          {onSignOut && (
            <button
              type="button"
              data-testid={DATA_TEST_ID.SIGN_OUT}
              onClick={onSignOut}
              className="mt-1.5 w-full rounded border border-cream/20 px-2.5 py-1.5 font-mono text-xs text-cream/75 hover:border-cream/50 hover:text-cream"
            >
              Sign out
            </button>
          )}
        </div>
      )}
    </aside>
  );
}
