import { forwardRef, type InputHTMLAttributes } from "react";
import { mergeClassNames } from "@/lib/mergeClassNames.ts";
import { Icon } from "@/components";

const uniqueId = "bd5abd68-85b0-455a-9d3e-f4efba44813d";

export const DATA_TEST_ID = {
  CONTAINER: `search-input-container-${uniqueId}`,
  INPUT: `search-input-input-${uniqueId}`,
};

export type SearchInputProps = InputHTMLAttributes<HTMLInputElement>;

export const SearchInput = forwardRef<HTMLInputElement, SearchInputProps>(
  function SearchInput({ className, ...rest }, ref) {
    return (
      <div
        data-testid={DATA_TEST_ID.CONTAINER}
        className={mergeClassNames("relative", className)}
      >
        <Icon
          name="search"
          size={14}
          className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted"
        />
        <input
          ref={ref}
          type="search"
          data-testid={DATA_TEST_ID.INPUT}
          className={mergeClassNames(
            "w-full rounded border bg-white py-2 pl-9 pr-3 text-[13px] text-ink outline-none transition-shadow",
            "border-line-strong",
            "focus:border-navy focus:ring-[3px] focus:ring-navy/10",
          )}
          {...rest}
        />
      </div>
    );
  },
);
