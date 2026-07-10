// Central component barrel — single entry point for cross-component imports.
// Per-component index.ts files were intentionally removed; everything is
// aggregated here so dead-code detection (knip) and tree-shaking work cleanly.

// AppLayout
export { AppLayout, DATA_TEST_ID as APP_LAYOUT_DATA_TEST_ID } from "./AppLayout/AppLayout";
export type { AppLayoutProps } from "./AppLayout/AppLayout";

// Breadcrumbs
export {
  Breadcrumbs,
  DATA_TEST_ID as BREADCRUMBS_DATA_TEST_ID,
} from "./Breadcrumbs/Breadcrumbs";
export type {
  BreadcrumbsProps,
  BreadcrumbItem,
} from "./Breadcrumbs/Breadcrumbs";

// Button
export { Button, DATA_TEST_ID as BUTTON_DATA_TEST_ID } from "./Button/Button";
export type { ButtonProps } from "./Button/Button.types";
export { buttonVariants } from "./Button/Button.variants";

// Card
export {
  Card,
  CardHead,
  DATA_TEST_ID as CARD_DATA_TEST_ID,
} from "./Card/Card";
export type { CardProps, CardHeadProps } from "./Card/Card";

// CodeBlock
export {
  CodeBlock,
  DATA_TEST_ID as CODE_BLOCK_DATA_TEST_ID,
} from "./CodeBlock/CodeBlock";
export type { CodeBlockProps } from "./CodeBlock/CodeBlock";

// Divider
export {
  Divider,
  DATA_TEST_ID as DIVIDER_DATA_TEST_ID,
} from "./Divider/Divider";
export type { DividerProps } from "./Divider/Divider";

// Drawer
export {
  Drawer,
  DATA_TEST_ID as DRAWER_DATA_TEST_ID,
} from "./Drawer/Drawer";
export type { DrawerProps } from "./Drawer/Drawer";

// EmptyState
export {
  EmptyState,
  DATA_TEST_ID as EMPTY_STATE_DATA_TEST_ID,
} from "./EmptyState/EmptyState";
export type { EmptyStateProps } from "./EmptyState/EmptyState";

// Eyebrow
export {
  Eyebrow,
  DATA_TEST_ID as EYEBROW_DATA_TEST_ID,
} from "./Eyebrow/Eyebrow";
export type { EyebrowProps } from "./Eyebrow/Eyebrow";

// FormField
export {
  FormField,
  DATA_TEST_ID as FORM_FIELD_DATA_TEST_ID,
} from "./FormField/FormField";
export type { FormFieldProps } from "./FormField/FormField";

// Icon
export { Icon, DATA_TEST_ID as ICON_DATA_TEST_ID } from "./Icon/Icon";
export type { IconName, IconProps } from "./Icon/Icon.types";

// IconButton
export {
  IconButton,
  DATA_TEST_ID as ICON_BUTTON_DATA_TEST_ID,
} from "./IconButton/IconButton";
export type { IconButtonProps } from "./IconButton/IconButton";

// Input
export { Input, DATA_TEST_ID as INPUT_DATA_TEST_ID } from "./Input/Input";
export type { InputProps } from "./Input/Input";

// Kbd
export { Kbd, DATA_TEST_ID as KBD_DATA_TEST_ID } from "./Kbd/Kbd";
export type { KbdProps } from "./Kbd/Kbd";

// Label
export { Label, DATA_TEST_ID as LABEL_DATA_TEST_ID } from "./Label/Label";
export type { LabelProps } from "./Label/Label";

// MethodBadge
export {
  MethodBadge,
  DATA_TEST_ID as METHOD_BADGE_DATA_TEST_ID,
} from "./MethodBadge/MethodBadge";
export type { HttpMethod, MethodBadgeProps } from "./MethodBadge/MethodBadge";

// Modal
export { Modal, DATA_TEST_ID as MODAL_DATA_TEST_ID } from "./Modal/Modal";
export type { ModalProps } from "./Modal/Modal";

// NavLink
export {
  NavLink,
  DATA_TEST_ID as NAV_LINK_DATA_TEST_ID,
} from "./NavLink/NavLink";
export type { NavLinkProps } from "./NavLink/NavLink";

// RadioCard
export {
  RadioCard,
  DATA_TEST_ID as RADIO_CARD_DATA_TEST_ID,
} from "./RadioCard/RadioCard";
export type { RadioCardProps } from "./RadioCard/RadioCard";

// ScoreBar
export {
  ScoreBar,
  DATA_TEST_ID as SCORE_BAR_DATA_TEST_ID,
} from "./ScoreBar/ScoreBar";
export type { ScoreBarProps } from "./ScoreBar/ScoreBar";

// SearchInput
export {
  SearchInput,
  DATA_TEST_ID as SEARCH_INPUT_DATA_TEST_ID,
} from "./SearchInput/SearchInput";
export type { SearchInputProps } from "./SearchInput/SearchInput";

// Select
export {
  Select,
  DATA_TEST_ID as SELECT_DATA_TEST_ID,
} from "./Select/Select";
export type { SelectProps } from "./Select/Select";

// Sidebar
export {
  Sidebar,
  DATA_TEST_ID as SIDEBAR_DATA_TEST_ID,
} from "./Sidebar/Sidebar";
export type { SidebarProps } from "./Sidebar/Sidebar";
export type {
  SidebarNavItem,
  SidebarNavGroup,
  SidebarUser,
} from "./Sidebar/Sidebar.types";

// Slider
export {
  Slider,
  DATA_TEST_ID as SLIDER_DATA_TEST_ID,
} from "./Slider/Slider";
export type { SliderProps } from "./Slider/Slider";

// Spinner
export {
  Spinner,
  DATA_TEST_ID as SPINNER_DATA_TEST_ID,
} from "./Spinner/Spinner";
export type { SpinnerProps } from "./Spinner/Spinner";

// StatusPill
export {
  StatusPill,
  DATA_TEST_ID as STATUS_PILL_DATA_TEST_ID,
} from "./StatusPill/StatusPill";
export type { StatusPillProps } from "./StatusPill/StatusPill";

// Table
export { Table, DATA_TEST_ID as TABLE_DATA_TEST_ID } from "./Table/Table";
export type { TableProps, TableRowProps } from "./Table/Table";

// Tabs
export { Tabs, DATA_TEST_ID as TABS_DATA_TEST_ID } from "./Tabs/Tabs";
export type { TabsProps, TabItem } from "./Tabs/Tabs";

// Tag
export { Tag, DATA_TEST_ID as TAG_DATA_TEST_ID } from "./Tag/Tag";
export type { TagProps } from "./Tag/Tag";
export { tagVariants } from "./Tag/Tag.variants";

// Textarea
export {
  Textarea,
  DATA_TEST_ID as TEXTAREA_DATA_TEST_ID,
} from "./Textarea/Textarea";
export type { TextareaProps } from "./Textarea/Textarea";

// Toast
export {
  ToastProvider,
  useToast,
  DATA_TEST_ID as TOAST_DATA_TEST_ID,
} from "./Toast/ToastProvider";
export type { ToastProviderProps } from "./Toast/ToastProvider";
export type {
  ToastContextValue,
  ToastInput,
  ToastItem,
  ToastPlacement,
  ToastTone,
} from "./Toast/Toast.types";

// Toggle
export {
  Toggle,
  DATA_TEST_ID as TOGGLE_DATA_TEST_ID,
} from "./Toggle/Toggle";
export type { ToggleProps } from "./Toggle/Toggle";

// Topbar
export {
  Topbar,
  DATA_TEST_ID as TOPBAR_DATA_TEST_ID,
} from "./Topbar/Topbar";
export type { TopbarProps } from "./Topbar/Topbar";

// StatCard
export {
  StatCard,
  DATA_TEST_ID as STAT_CARD_DATA_TEST_ID,
} from "./StatCard/StatCard";
export type { StatCardProps } from "./StatCard/StatCard";

// UsageChart
export {
  UsageChart,
  DATA_TEST_ID as USAGE_CHART_DATA_TEST_ID,
} from "./UsageChart/UsageChart";
export type { UsageChartProps } from "./UsageChart/UsageChart";
