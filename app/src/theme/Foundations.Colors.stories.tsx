import type { Meta, StoryObj } from "@storybook/react";
import { colors, ENTITY_TYPES } from "./theme";

const meta: Meta = {
  title: "Foundations/Colors",
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "All brand color tokens. Mirrors the CSS variables in the design handoff styles.css. Tailwind utility classes derive from these — `bg-navy`, `text-lime`, `border-line-strong`, etc.",
      },
    },
  },
};
export default meta;

type Story = StoryObj;

function Swatch({ name, value }: { name: string; value: string }) {
  return (
    <div className="flex items-center gap-3 text-xs">
      <div
        className="h-10 w-10 rounded border border-line-strong"
        style={{ background: value }}
      />
      <div className="font-mono">
        <div className="font-medium text-ink">{name}</div>
        <div className="text-muted">{value}</div>
      </div>
    </div>
  );
}

function flatten(
  obj: Record<string, unknown>,
  prefix = "",
): { name: string; value: string }[] {
  const out: { name: string; value: string }[] = [];
  for (const [k, v] of Object.entries(obj)) {
    const name = k === "DEFAULT" ? prefix.replace(/\.$/, "") : `${prefix}${k}`;
    if (typeof v === "string") {
      out.push({ name, value: v });
    } else if (v && typeof v === "object") {
      out.push(...flatten(v as Record<string, unknown>, `${name}.`));
    }
  }
  return out;
}

export const Brand: Story = {
  render: () => {
    const palette = flatten({
      navy: colors.navy,
      ink: colors.ink,
      lime: colors.lime,
      yellow: colors.yellow,
      teal: colors.teal,
      cream: colors.cream,
      paper: colors.paper,
      line: colors.line,
      muted: colors.muted,
      error: colors.error,
    });
    return (
      <div className="grid grid-cols-3 gap-5">
        {palette.map((c) => (
          <Swatch key={c.name} name={c.name} value={c.value} />
        ))}
      </div>
    );
  },
};

export const Entities: Story = {
  render: () => (
    <div className="grid grid-cols-2 gap-6">
      {ENTITY_TYPES.map((type) => {
        const t = colors.entity[type];
        return (
          <div
            key={type}
            className="rounded-md border border-line bg-paper p-4"
          >
            <div className="mb-3 flex items-baseline justify-between">
              <span className="font-mono text-sm font-medium text-navy">
                {type}
              </span>
              <span className="font-mono text-xs text-muted">
                fg {t.fg} · bg {t.bg}
              </span>
            </div>
            <div className="space-y-3">
              <p className="serif text-base leading-relaxed">
                Looking for a{" "}
                <span className="ent" data-type={type}>
                  sample entity
                </span>{" "}
                in this sentence.
              </p>
              <div className="flex gap-2">
                <div
                  className="h-5 w-5 rounded-sm"
                  style={{ background: t.fg }}
                />
                <div
                  className="h-5 w-5 rounded-sm"
                  style={{ background: t.bg }}
                />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  ),
};
