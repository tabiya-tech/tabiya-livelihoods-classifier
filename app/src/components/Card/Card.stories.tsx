import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { Card, CardHead } from "./Card";
import { Button } from "@/components";
import { Eyebrow } from "@/components";

const meta: Meta<typeof Card> = {
  title: "Primitives/Card",
  component: Card,
  parameters: { layout: "padded" },
};
export default meta;

type Story = StoryObj<typeof Card>;

export const Plain: Story = {
  render: () => (
    <Card className="max-w-md">
      <Eyebrow>Calls this month</Eyebrow>
      <div className="mt-2 font-mono text-3xl text-navy">12,348</div>
    </Card>
  ),
};

export const WithHead: Story = {
  render: () => (
    <Card className="max-w-md">
      <CardHead
        title="Recent classifications"
        action={
          <Button size="sm" variant="ghost" onClick={fn()}>
            View all
          </Button>
        }
      />
      <p className="text-sm text-muted">Table content goes here.</p>
    </Card>
  ),
};

export const Flush: Story = {
  render: () => (
    <Card flush className="max-w-md">
      <div className="px-4 py-3 font-mono text-xs text-muted">Flush card – host owns padding</div>
    </Card>
  ),
};

export const Elevated: Story = {
  render: () => (
    <Card elevated className="max-w-md">
      <Eyebrow>Elevated</Eyebrow>
      <p className="text-sm">Used for overlays or featured surfaces.</p>
    </Card>
  ),
};
