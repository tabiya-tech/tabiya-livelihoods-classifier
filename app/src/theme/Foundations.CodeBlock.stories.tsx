import type { Meta, StoryObj } from "@storybook/react";

const meta: Meta = {
  title: "Foundations/Code Block",
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Signature dark code block. Use the `.code-block` class on a `<pre>`. Inline tokens get `.ck-key`, `.ck-str`, `.ck-num`, or `.ck-com`.",
      },
    },
  },
};
export default meta;

type Story = StoryObj;

export const Default: Story = {
  render: () => (
    <pre className="code-block">
      {`curl -X POST https://api.classifier.tabiya.tech/v1/classify \\
  -H "x-api-key: $TABIYA_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"text":"Senior Data Engineer with Python."}'`}
    </pre>
  ),
};

export const Muted: Story = {
  render: () => (
    <pre className="code-block muted">
      {`{
  "status": "healthy",
  "service": "classify-api",
  "dependencies": {
    "ner_api": "healthy",
    "nel_api": "healthy"
  }
}`}
    </pre>
  ),
};

export const Tokenized: Story = {
  render: () => (
    <pre className="code-block">
      {"{\n  "}
      <span className="ck-key">&quot;status&quot;</span>: <span className="ck-str">&quot;healthy&quot;</span>,{"\n  "}
      <span className="ck-key">&quot;processed&quot;</span>: <span className="ck-num">1</span>,{"\n  "}
      <span className="ck-com">{"// per-dependency health"}</span>{"\n  "}
      <span className="ck-key">&quot;dependencies&quot;</span>: {"{ ... }"}
      {"\n}"}
    </pre>
  ),
};
