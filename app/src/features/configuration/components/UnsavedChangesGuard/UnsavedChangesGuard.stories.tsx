import type { Meta, StoryObj } from "@storybook/react";
import { useState } from "react";
import { fn } from "@storybook/test";
import { Button } from "@/components";
import { UnsavedChangesGuard } from "./UnsavedChangesGuard";

const meta: Meta<typeof UnsavedChangesGuard> = {
  title: "Features/Configuration/UnsavedChangesGuard",
  component: UnsavedChangesGuard,
  parameters: { layout: "padded" },
  args: {
    onConfirm: fn(),
    onCancel: fn(),
  },
};
export default meta;

type Story = StoryObj<typeof UnsavedChangesGuard>;

export const Open: Story = {
  args: { open: true },
};

export const Closed: Story = {
  args: { open: false },
};

/**
 * Interactive harness — opens/closes the guard via a trigger button so that
 * the AnimatePresence transitions can be exercised.
 */
export const Interactive: Story = {
  render: function InteractiveStory(args) {
    const [open, setOpen] = useState(false);
    return (
      <div className="flex flex-col gap-3">
        <Button onClick={() => setOpen(true)}>Try to leave the page</Button>
        <UnsavedChangesGuard
          {...args}
          open={open}
          onConfirm={() => {
            args.onConfirm();
            setOpen(false);
          }}
          onCancel={() => {
            args.onCancel();
            setOpen(false);
          }}
        />
      </div>
    );
  },
};
