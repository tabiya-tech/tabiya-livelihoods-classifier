import type { Meta, StoryObj } from "@storybook/react";
import { ValidationBadge } from "./ValidationBadge";

const meta: Meta<typeof ValidationBadge> = {
  title: "Features/Pipelines/ValidationBadge",
  component: ValidationBadge,
  parameters: { layout: "centered" },
  args: { count: 2 },
};
export default meta;

type Story = StoryObj<typeof ValidationBadge>;

export const ErrorSeverity: Story = {
  args: { severity: "error", count: 3, title: "3 validation errors" },
};

export const WarningSeverity: Story = {
  args: { severity: "warning", count: 1, title: "1 warning" },
};

export const InfoSeverity: Story = {
  args: { severity: "info", count: 5, title: "5 info messages" },
};
