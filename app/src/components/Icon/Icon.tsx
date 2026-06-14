import type { IconName, IconProps } from "./Icon.types";

const uniqueId = "5789bcfb-a547-4739-9446-d67a89d8c368";

export const DATA_TEST_ID = {
  SVG: `icon-svg-${uniqueId}`,
};

const paths: Record<IconName, JSX.Element> = {
  classify: (
    <>
      <path d="M2 3h12M2 8h8M2 13h12" />
      <circle cx="13" cy="8" r="1.5" fill="currentColor" stroke="none" />
    </>
  ),
  dashboard: (
    <>
      <rect x="2" y="2" width="5" height="6" rx="1" />
      <rect x="9" y="2" width="5" height="3" rx="1" />
      <rect x="9" y="7" width="5" height="7" rx="1" />
      <rect x="2" y="10" width="5" height="4" rx="1" />
    </>
  ),
  config: (
    <>
      <circle cx="8" cy="8" r="2" />
      <path d="M8 1.5v2M8 12.5v2M14.5 8h-2M3.5 8h-2M12.6 3.4l-1.4 1.4M4.8 11.2l-1.4 1.4M12.6 12.6l-1.4-1.4M4.8 4.8L3.4 3.4" />
    </>
  ),
  key: (
    <>
      <circle cx="5" cy="11" r="3" />
      <path d="M7 9l6-6M11 5l2 2" />
    </>
  ),
  docs: (
    <>
      <path d="M3 2.5h7l3 3V13a.5.5 0 01-.5.5h-9A.5.5 0 013 13V3a.5.5 0 010-.5z" />
      <path d="M10 2.5V6h3M5.5 8.5h5M5.5 11h5" />
    </>
  ),
  history: (
    <>
      <path d="M2.5 5A6 6 0 1014 8" />
      <path d="M2.5 2v3h3M8 4.5V8l2.5 1.5" />
    </>
  ),
  copy: (
    <>
      <rect x="5" y="5" width="9" height="9" rx="1.5" />
      <path d="M2 11V3a1 1 0 011-1h8" />
    </>
  ),
  arrowRight: <path d="M3 8h10M9 4l4 4-4 4" />,
  external: (
    <path d="M9 2h5v5M14 2L7.5 8.5M12 9v3.5a.5.5 0 01-.5.5h-8A.5.5 0 013 12.5v-8a.5.5 0 01.5-.5H7" />
  ),
  plus: <path d="M8 3v10M3 8h10" strokeWidth={1.6} />,
  trash: (
    <path d="M3 4.5h10M6 4V3a.5.5 0 01.5-.5h3A.5.5 0 0110 3v1M5 4.5l.5 8.5a.5.5 0 00.5.5h4a.5.5 0 00.5-.5L11 4.5" />
  ),
  check: <path d="M3 8.5l3 3 7-7" strokeWidth={1.6} />,
  close: <path d="M3.5 3.5l9 9M12.5 3.5l-9 9" />,
  filter: <path d="M2 3h12l-4.5 6V13l-3 1.5V9L2 3z" />,
  download: <path d="M8 2v8M4.5 7.5L8 11l3.5-3.5M3 13.5h10" />,
  upload: <path d="M8 11V3M4.5 6.5L8 3l3.5 3.5M3 13.5h10" />,
  search: (
    <>
      <circle cx="7" cy="7" r="4.5" />
      <path d="M10.5 10.5l3 3" />
    </>
  ),
  spark: (
    <path
      d="M8 1.5l1.6 4.4 4.4 1.6-4.4 1.6L8 13.5l-1.6-4.4L2 7.5l4.4-1.6L8 1.5z"
      fill="currentColor"
      stroke="none"
    />
  ),
};

export function Icon({ name, size = 14, ...rest }: IconProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.4}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      data-testid={DATA_TEST_ID.SVG}
      data-icon={name}
      {...rest}
    >
      {paths[name]}
    </svg>
  );
}
