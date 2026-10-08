import type { Meta, StoryObj } from "@storybook/react";
import { Icon } from "./Icon";
import type { IconName } from "./Icon.types";

const meta: Meta<typeof Icon> = {
  title: "Primitives/Icon",
  component: Icon,
  parameters: { layout: "padded" },
  args: { size: 24 },
};
export default meta;

type Story = StoryObj<typeof Icon>;

const allNames: IconName[] = [
  "classify",
  "dashboard",
  "config",
  "key",
  "docs",
  "history",
  "copy",
  "arrowRight",
  "external",
  "plus",
  "trash",
  "check",
  "close",
  "filter",
  "download",
  "upload",
  "search",
  "spark",
  "globe",
];

export const Gallery: Story = {
  render: (args) => (
    <div className="grid grid-cols-4 gap-4 text-navy">
      {allNames.map((name) => (
        <div
          key={name}
          className="flex items-center gap-3 rounded-md border border-line bg-paper p-3"
        >
          <Icon {...args} name={name} />
          <span className="font-mono text-xs text-muted">{name}</span>
        </div>
      ))}
    </div>
  ),
};

export const Single: Story = {
  args: { name: "arrowRight", size: 32 },
};
