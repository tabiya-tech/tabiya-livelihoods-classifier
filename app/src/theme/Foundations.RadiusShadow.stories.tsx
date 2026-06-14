import type { Meta, StoryObj } from "@storybook/react";
import { borderRadius, boxShadow } from "./theme";

const meta: Meta = {
  title: "Foundations/Radius & Shadow",
  parameters: { layout: "padded" },
};
export default meta;

type Story = StoryObj;

export const Radius: Story = {
  render: () => (
    <div className="grid grid-cols-4 gap-5">
      {Object.entries(borderRadius).map(([name, value]) => (
        <div key={name} className="space-y-2">
          <div
            className="h-20 w-full border border-line-strong bg-paper"
            style={{ borderRadius: value as string }}
          />
          <div className="font-mono text-xs">
            <div className="font-medium text-ink">
              rounded{name === "DEFAULT" ? "" : `-${name}`}
            </div>
            <div className="text-muted">{value as string}</div>
          </div>
        </div>
      ))}
    </div>
  ),
};

export const Shadow: Story = {
  render: () => (
    <div className="grid grid-cols-2 gap-6 bg-cream p-6">
      {Object.entries(boxShadow).map(([name, value]) => (
        <div key={name} className="space-y-2">
          <div
            className="h-24 w-full rounded-md border border-line bg-paper"
            style={{ boxShadow: value as string }}
          />
          <div className="font-mono text-xs">
            <div className="font-medium text-ink">shadow-{name}</div>
            <div className="text-muted">{value as string}</div>
          </div>
        </div>
      ))}
    </div>
  ),
};
