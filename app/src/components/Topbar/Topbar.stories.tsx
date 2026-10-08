import type { Meta, StoryObj } from "@storybook/react";
import { Topbar } from "./Topbar";
import { StatusPill } from "@/components";
import { Kbd } from "@/components";

const meta: Meta<typeof Topbar> = {
  title: "Primitives/Topbar",
  component: Topbar,
  parameters: { layout: "fullscreen" },
};
export default meta;

type Story = StoryObj<typeof Topbar>;

export const Default: Story = {
  render: () => (
    <Topbar
      breadcrumbs={[
        { label: "Workspace", onClick: () => {} },
        { label: "Classifier" },
      ]}
      right={
        <>
          <StatusPill status="healthy">API healthy · v1.0.0</StatusPill>
          <Kbd>⌘ K</Kbd>
        </>
      }
    />
  ),
};
