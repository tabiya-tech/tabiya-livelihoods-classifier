import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { FormField } from "./FormField";
import { Input } from "@/components";
import { Textarea } from "@/components";

const meta: Meta<typeof FormField> = {
  title: "Primitives/FormField",
  component: FormField,
  parameters: { layout: "padded" },
};
export default meta;

type Story = StoryObj<typeof FormField>;

export const Default: Story = {
  render: () => (
    <FormField label="Key label" help="Use a name that's easy to recognize.">
      <Input mono placeholder="Production · staging · my notebook" onChange={fn()} />
    </FormField>
  ),
};

export const Required: Story = {
  render: () => (
    <FormField label="Email" required>
      <Input type="email" placeholder="you@tabiya.org" onChange={fn()} />
    </FormField>
  ),
};

export const WithError: Story = {
  render: () => (
    <FormField label="API key" error="This key doesn't match an OpenAI prefix.">
      <Input mono defaultValue="abc123" onChange={fn()} />
    </FormField>
  ),
};

export const Textareaish: Story = {
  render: () => (
    <FormField label="Source text" help="Paste a full job description.">
      <Textarea
        rows={6}
        placeholder="Looking for a Senior Data Engineer…"
        onChange={fn()}
      />
    </FormField>
  ),
};
