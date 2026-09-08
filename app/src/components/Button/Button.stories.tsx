import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { Button } from "./Button";
import { Icon } from "@/components";

const meta: Meta<typeof Button> = {
  title: "Primitives/Button",
  component: Button,
  parameters: { layout: "centered" },
  argTypes: {
    variant: {
      control: "select",
      options: ["default", "primary", "lime", "ghost", "danger"],
    },
    size: { control: "select", options: ["sm", "md", "lg"] },
  },
  args: { children: "Run classify", onClick: fn() },
};
export default meta;

type Story = StoryObj<typeof Button>;

export const Default: Story = {};

export const Primary: Story = { args: { variant: "primary" } };
export const Lime: Story = { args: { variant: "lime", children: "Open the Classifier" } };
export const Ghost: Story = { args: { variant: "ghost" } };
export const Danger: Story = { args: { variant: "danger", children: "Revoke" } };

export const Small: Story = { args: { size: "sm" } };
export const Large: Story = { args: { size: "lg", variant: "primary" } };

export const WithIcons: Story = {
  args: {
    variant: "primary",
    leading: <Icon name="plus" />,
    trailing: <Icon name="arrowRight" />,
  },
};

export const Loading: Story = {
  args: { variant: "primary", loading: true, children: "Running" },
};

export const Disabled: Story = {
  args: { variant: "primary", disabled: true },
};

export const AllVariants: Story = {
  args: { onClick: fn() },
  render: (args) => (
    <div className="flex flex-wrap items-center gap-2">
      <Button {...args} variant="default">Default</Button>
      <Button {...args} variant="primary">Primary</Button>
      <Button {...args} variant="lime">Lime</Button>
      <Button {...args} variant="ghost">Ghost</Button>
      <Button {...args} variant="danger">Danger</Button>
    </div>
  ),
};
