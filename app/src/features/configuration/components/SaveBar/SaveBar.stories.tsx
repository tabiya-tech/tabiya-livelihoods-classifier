import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { SaveBar } from "./SaveBar";

const meta: Meta<typeof SaveBar> = {
  title: "Features/Configuration/SaveBar",
  component: SaveBar,
  parameters: {
    layout: "fullscreen",
  },
  decorators: [
    function SaveBarFrame(StoryComponent) {
      return (
        <div className="relative h-[320px] w-full bg-paper">
          <div className="px-8 py-6 text-sm text-muted">
            Page content sits here. The save bar docks to the bottom edge of the
            frame.
          </div>
          <StoryComponent />
        </div>
      );
    },
  ],
  args: {
    onSave: fn(),
    onDiscard: fn(),
    leftOffset: 0,
  },
};
export default meta;

type Story = StoryObj<typeof SaveBar>;

export const Dirty: Story = {
  args: {
    isDirty: true,
    saveStatus: "idle",
  },
};

export const Saving: Story = {
  args: {
    isDirty: true,
    saveStatus: "saving",
  },
};

export const Saved: Story = {
  args: {
    isDirty: false,
    saveStatus: "saved",
  },
};

/**
 * When the draft is clean and no save is in flight the bar is unmounted.
 * The story renders nothing under the page content as a result.
 */
export const Hidden: Story = {
  args: {
    isDirty: false,
    saveStatus: "idle",
  },
};
