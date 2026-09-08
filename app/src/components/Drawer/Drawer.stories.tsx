import type { Meta, StoryObj } from "@storybook/react";
import { useState } from "react";
import { fn } from "@storybook/test";
import { Drawer } from "./Drawer";
import { Button } from "@/components";

const meta: Meta<typeof Drawer> = {
  title: "Primitives/Drawer",
  component: Drawer,
  parameters: { layout: "centered" },
};
export default meta;

type Story = StoryObj<typeof Drawer>;

export const EntityDetail: Story = {
  render: () => {
    const [isOpen, setIsOpen] = useState(false);
    const onOpen = fn(() => setIsOpen(true));
    const onClose = fn(() => setIsOpen(false));
    const onViewInEsco = fn();
    return (
      <>
        <Button onClick={onOpen}>Open drawer</Button>
        <Drawer
          open={isOpen}
          onClose={onClose}
          eyebrow="occupation"
          title="Senior Data Engineer"
          description="span [0, 20]"
          footer={
            <div className="flex justify-end gap-2">
              <Button variant="ghost" onClick={onClose}>
                Close
              </Button>
              <Button variant="primary" onClick={onViewInEsco}>
                View in ESCO
              </Button>
            </div>
          }
        >
          <div className="space-y-3 text-sm text-muted">
            <p>Top ESCO match: data engineer (94%)</p>
            <p>Alt match: big data engineer (88%)</p>
          </div>
        </Drawer>
      </>
    );
  },
};
