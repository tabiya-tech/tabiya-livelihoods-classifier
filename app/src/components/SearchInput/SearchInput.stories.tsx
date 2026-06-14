import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { SearchInput } from "./SearchInput";

const meta: Meta<typeof SearchInput> = {
  title: "Primitives/SearchInput",
  component: SearchInput,
  parameters: { layout: "padded" },
  args: { placeholder: "Search by title, entity, or token…", onChange: fn() },
};
export default meta;

type Story = StoryObj<typeof SearchInput>;

export const Default: Story = {
  render: (args) => (
    <div className="max-w-md">
      <SearchInput {...args} />
    </div>
  ),
};
