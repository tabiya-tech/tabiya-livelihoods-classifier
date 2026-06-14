import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { IconButton } from "./IconButton";

const meta: Meta<typeof IconButton> = {
  title: "Primitives/IconButton",
  component: IconButton,
  parameters: { layout: "centered" },
  argTypes: {
    variant: { control: "select", options: ["default", "ghost", "primary", "danger"] },
    size: { control: "select", options: ["sm", "md"] },
  },
  args: { icon: "copy", "aria-label": "Copy", onClick: fn() },
};
export default meta;

type Story = StoryObj<typeof IconButton>;

export const Default: Story = {};
export const Primary: Story = { args: { variant: "primary", icon: "arrowRight", "aria-label": "Next" } };
export const Danger: Story = { args: { variant: "danger", icon: "trash", "aria-label": "Delete" } };

export const Cluster: Story = {
  render: () => {
    const onClick = fn();
    return (
      <div className="flex items-center gap-2">
        <IconButton icon="copy" aria-label="Copy" onClick={onClick} />
        <IconButton icon="download" aria-label="Download" onClick={onClick} />
        <IconButton icon="upload" aria-label="Upload" onClick={onClick} />
        <IconButton icon="trash" aria-label="Delete" variant="danger" onClick={onClick} />
      </div>
    );
  },
};
