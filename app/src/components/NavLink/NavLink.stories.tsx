import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { NavLink } from "./NavLink";
import { Icon } from "@/components";

const meta: Meta<typeof NavLink> = {
  title: "Primitives/NavLink",
  component: NavLink,
  parameters: { layout: "centered" },
  decorators: [
    (Story) => (
      <div className="bg-navy p-4">
        <Story />
      </div>
    ),
  ],
  args: { onClick: fn() },
};
export default meta;

type Story = StoryObj<typeof NavLink>;

export const Inactive: Story = {
  args: { children: "Classifier", icon: <Icon name="classify" /> },
};

export const Active: Story = {
  args: { active: true, children: "Classifier", icon: <Icon name="classify" /> },
};

export const Stack: Story = {
  render: () => {
    const onClick = fn();
    return (
      <div className="flex w-44 flex-col gap-0.5">
        <NavLink active icon={<Icon name="classify" />} onClick={onClick}>
          Classifier
        </NavLink>
        <NavLink icon={<Icon name="dashboard" />} onClick={onClick}>
          Dashboard
        </NavLink>
        <NavLink icon={<Icon name="history" />} onClick={onClick}>
          History
        </NavLink>
      </div>
    );
  },
};
