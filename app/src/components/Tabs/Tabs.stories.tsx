import type { Meta, StoryObj } from "@storybook/react";
import { useState } from "react";
import { fn } from "@storybook/test";
import { Tabs } from "./Tabs";

const meta: Meta<typeof Tabs> = {
  title: "Primitives/Tabs",
  component: Tabs,
  parameters: { layout: "padded" },
};
export default meta;

type Story = StoryObj<typeof Tabs>;

export const Default: Story = {
  render: () => {
    const [activeTab, setActiveTab] = useState("annotated");
    const onChange = fn((tabId: string) => setActiveTab(tabId));
    return (
      <div>
        <Tabs
          aria-label="Results view"
          value={activeTab}
          onChange={onChange}
          items={[
            { id: "annotated", label: "Entities", meta: "21" },
            { id: "table", label: "Table" },
            { id: "json", label: "JSON" },
            { id: "disabled", label: "Soon", disabled: true },
          ]}
        />
        <div className="mt-6 font-mono text-sm text-muted">
          Active panel: <span className="text-navy">{activeTab}</span>
        </div>
      </div>
    );
  },
};
