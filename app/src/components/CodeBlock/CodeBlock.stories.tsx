import type { Meta, StoryObj } from "@storybook/react";
import { CodeBlock } from "./CodeBlock";

const meta: Meta<typeof CodeBlock> = {
  title: "Primitives/CodeBlock",
  component: CodeBlock,
  parameters: { layout: "padded" },
};
export default meta;

type Story = StoryObj<typeof CodeBlock>;

const curl = `curl -X POST https://api.classifier.tabiya.tech/v1/classify \\
  -H "x-api-key: $TABIYA_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"text":"Senior Data Engineer with Python."}'`;

const response = `{
  "status": "healthy",
  "service": "classify-api",
  "dependencies": { "ner_api": "healthy", "nel_api": "healthy" }
}`;

export const Default: Story = {
  args: { code: curl },
};

export const Muted: Story = {
  args: { code: response, muted: true },
};

export const Copyable: Story = {
  args: { code: curl, copyable: true },
};

export const Scrolling: Story = {
  args: {
    copyable: true,
    maxHeight: 180,
    code: Array.from({ length: 30 }, (_, i) => `line ${i + 1}: payload data ${i}`).join("\n"),
  },
};
