import type { Meta, StoryObj } from "@storybook/react";
import { useState } from "react";
import { fn } from "@storybook/test";
import { Toggle } from "./Toggle";

const meta: Meta<typeof Toggle> = {
  title: "Primitives/Toggle",
  component: Toggle,
  parameters: { layout: "centered" },
  args: { onChange: fn() },
};
export default meta;

type Story = StoryObj<typeof Toggle>;

export const Uncontrolled: Story = {
  args: { defaultChecked: false, label: "Show line numbers" },
};

export const Controlled: Story = {
  render: () => {
    const [isOn, setIsOn] = useState(true);
    const onChange = fn((nextValue: boolean) => setIsOn(nextValue));
    return (
      <div className="flex items-center gap-3">
        <Toggle checked={isOn} onChange={onChange} label="Auto-classify on paste" />
        <span className="font-mono text-xs text-muted">{isOn ? "on" : "off"}</span>
      </div>
    );
  },
};

export const Disabled: Story = {
  args: { defaultChecked: true, disabled: true, label: "Locked toggle" },
};
