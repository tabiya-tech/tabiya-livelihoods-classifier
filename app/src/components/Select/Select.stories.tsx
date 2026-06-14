import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { Select } from "./Select";

const meta: Meta<typeof Select> = {
  title: "Primitives/Select",
  component: Select,
  parameters: { layout: "padded" },
  args: { onChange: fn() },
};
export default meta;

type Story = StoryObj<typeof Select>;

export const Default: Story = {
  render: (args) => (
    <Select {...args} defaultValue="esco-1.2.0">
      <option value="esco-1.1.1">ESCO v1.1.1</option>
      <option value="esco-1.2.0">ESCO v1.2.0</option>
      <option value="isco-08">ISCO 08</option>
    </Select>
  ),
};

export const Mono: Story = {
  args: { mono: true },
  render: (args) => (
    <Select {...args}>
      <option>tabiya/roberta-base-job-ner</option>
      <option>tabiya/deberta-v3-job-ner</option>
      <option>tabiya/distilbert-job-ner</option>
    </Select>
  ),
};

export const Invalid: Story = {
  args: { invalid: true },
  render: (args) => (
    <Select {...args}>
      <option>Pick a model</option>
    </Select>
  ),
};
