import { Icon } from "@/components/Icon/Icon";
import { mergeClassNames } from "@/lib/mergeClassNames";
import type { IconName } from "@/components/Icon/Icon.types";

const uniqueId = "f3e2d1c0-b9a8-4f7e-8d6c-5b4a3c2d1e0f";

export const DATA_TEST_ID = {
  CONTAINER: `bottom-tab-bar-container-${uniqueId}`,
  TAB: `bottom-tab-bar-tab-${uniqueId}`,
};

export interface BottomTabItem {
  id: string;
  label: string;
  /** Shorter label for narrow viewports. Falls back to `label` if omitted. */
  shortLabel?: string;
  icon: IconName;
}

export interface BottomTabBarProps {
  items: BottomTabItem[];
  activeId: string;
  onNavigate: (id: string) => void;
  className?: string;
}

export function BottomTabBar({
  items,
  activeId,
  onNavigate,
  className,
}: BottomTabBarProps) {
  return (
    <nav
      data-testid={DATA_TEST_ID.CONTAINER}
      className={mergeClassNames(
        "fixed bottom-0 left-0 right-0 z-40 flex h-16 items-stretch border-t border-line bg-navy md:hidden",
        className,
      )}
    >
      {items.map((item) => {
        const isActive = item.id === activeId;
        return (
          <button
            key={item.id}
            type="button"
            data-testid={DATA_TEST_ID.TAB}
            data-nav-id={item.id}
            onClick={() => onNavigate(item.id)}
            className={mergeClassNames(
              "flex flex-1 flex-col items-center justify-center gap-0.5 text-[10px] font-mono tracking-wide transition-colors",
              isActive ? "text-lime" : "text-cream/50 hover:text-cream/80",
            )}
          >
            <Icon name={item.icon} size={18} />
            <span className="w-full truncate text-center">{item.shortLabel ?? item.label}</span>
          </button>
        );
      })}
    </nav>
  );
}
