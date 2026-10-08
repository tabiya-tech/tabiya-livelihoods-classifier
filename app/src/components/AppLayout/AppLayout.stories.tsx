import type { Meta, StoryObj } from "@storybook/react";
import { useState } from "react";
import { AppLayout } from "./AppLayout";
import { Sidebar } from "@/components";
import { Topbar } from "@/components";
import { StatusPill } from "@/components";
import { Kbd } from "@/components";

const meta: Meta<typeof AppLayout> = {
  title: "Primitives/AppLayout",
  component: AppLayout,
  parameters: { layout: "fullscreen" },
};
export default meta;

type Story = StoryObj<typeof AppLayout>;

const groups = [
  {
    label: "Workspace",
    items: [
      { id: "classify", label: "Classifier", icon: "classify" as const },
      { id: "dashboard", label: "Dashboard", icon: "dashboard" as const },
      { id: "history", label: "History", icon: "history" as const },
    ],
  },
  {
    label: "Settings",
    items: [
      { id: "config", label: "Configuration", icon: "config" as const },
      { id: "keys", label: "Keys", icon: "key" as const },
      { id: "docs", label: "Documentation", icon: "docs" as const },
    ],
  },
];

export const Shell: Story = {
  render: () => {
    const [active, setActive] = useState("classify");
    return (
      <AppLayout
        sidebar={
          <Sidebar
            activeId={active}
            onNavigate={setActive}
            groups={groups}
            user={{ initials: "SM", label: "sara.m@tabiya.org" }}
            onSignOut={() => {}}
          />
        }
        topbar={
          <Topbar
            breadcrumbs={[
              { label: "Workspace", onClick: () => {} },
              { label: active },
            ]}
            right={
              <>
                <StatusPill status="healthy">API healthy · v1.0.0</StatusPill>
                <Kbd>⌘ K</Kbd>
              </>
            }
          />
        }
      >
        <div className="px-10 py-8">
          <div className="eyebrow mb-2">Workspace · {active}</div>
          <h1 className="h-page">Page content goes here</h1>
        </div>
      </AppLayout>
    );
  },
};
