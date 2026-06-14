import type { Meta, StoryObj } from "@storybook/react";

const meta: Meta = {
  title: "Foundations/Typography",
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Editorial type scale. IBM Plex Mono carries the voice; Inter handles body; Source Serif 4 sets long-form reading. Use `.eyebrow`, `.h-display`, `.h-page`, `.h-section` for headings and `font-mono`/`font-sans`/`font-serif` for everything else.",
      },
    },
  },
};
export default meta;

type Story = StoryObj;

export const Scale: Story = {
  render: () => (
    <div className="space-y-8">
      <section>
        <div className="eyebrow mb-2">eyebrow · 11 / mono / uppercase</div>
        <h1 className="h-display">h-display · 32 / mono</h1>
      </section>
      <section>
        <div className="eyebrow mb-2">h-page · 24 / mono</div>
        <h2 className="h-page">Extract entities from a job description</h2>
      </section>
      <section>
        <div className="eyebrow mb-2">h-section · 14 / mono</div>
        <h3 className="h-section">Pipeline parameters</h3>
      </section>
      <section>
        <div className="eyebrow mb-2">body · 14 / sans</div>
        <p className="max-w-prose text-ink">
          Paste or upload text below. The pipeline runs in two stages — NER
          detects entity spans, then NEL links each one to the ESCO taxonomy.
        </p>
      </section>
      <section>
        <div className="eyebrow mb-2">lede · 16 / serif</div>
        <p className="max-w-prose font-serif text-base leading-relaxed text-muted">
          Every request to the Classifier API must carry a valid API key in the{" "}
          <code className="font-mono">x-api-key</code> header.
        </p>
      </section>
      <section>
        <div className="eyebrow mb-2">mono inline</div>
        <p className="text-ink">
          Set <code className="font-mono text-navy">top_k = 3</code> and{" "}
          <code className="font-mono text-navy">min_similarity = 0.5</code>.
        </p>
      </section>
    </div>
  ),
};

export const Families: Story = {
  render: () => (
    <div className="space-y-6">
      <Row label="font-mono · IBM Plex Mono" className="font-mono" />
      <Row label="font-sans · Inter" className="font-sans" />
      <Row label="font-serif · Source Serif 4" className="font-serif" />
    </div>
  ),
};

function Row({ label, className }: { label: string; className: string }) {
  return (
    <div>
      <div className="eyebrow mb-1">{label}</div>
      <p className={`${className} text-2xl text-navy`}>
        The quick brown fox jumps over the lazy dog · 0123456789
      </p>
    </div>
  );
}
