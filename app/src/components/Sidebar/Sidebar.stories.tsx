import type { Meta, StoryObj } from "@storybook/react";
import { useState } from "react";
import { fn } from "@storybook/test";
import { Sidebar } from "./Sidebar";

const meta: Meta<typeof Sidebar> = {
  title: "Primitives/Sidebar",
  component: Sidebar,
  parameters: { layout: "fullscreen" },
};
export default meta;

type Story = StoryObj<typeof Sidebar>;

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

export const Default: Story = {
  render: () => {
    const [activeRouteId, setActiveRouteId] = useState("classify");
    const onNavigate = fn((routeId: string) => setActiveRouteId(routeId));
    const onBrandClick = fn();
    const onSignOut = fn();
    return (
      <div className="flex min-h-screen bg-cream">
        <Sidebar
          activeId={activeRouteId}
          onNavigate={onNavigate}
          onBrandClick={onBrandClick}
          groups={groups}
          user={{ initials: "SM", label: "sara.m@tabiya.org" }}
          onSignOut={onSignOut}
        />
      </div>
    );
  },
};
