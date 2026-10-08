import type { Meta, StoryObj } from "@storybook/react";
import { useState } from "react";
import { fn } from "@storybook/test";
import { Modal } from "./Modal";
import { Button } from "@/components";

const meta: Meta<typeof Modal> = {
  title: "Primitives/Modal",
  component: Modal,
  parameters: { layout: "centered" },
};
export default meta;

type Story = StoryObj<typeof Modal>;

export const Confirmation: Story = {
  render: () => {
    const [isOpen, setIsOpen] = useState(false);
    const onOpen = fn(() => setIsOpen(true));
    const onClose = fn(() => setIsOpen(false));
    const onConfirm = fn(() => setIsOpen(false));
    return (
      <>
        <Button variant="danger" onClick={onOpen}>
          Revoke key
        </Button>
        <Modal
          open={isOpen}
          onClose={onClose}
          title="Revoke API key?"
          description="Any service using this key will start receiving 401 responses."
          footer={
            <>
              <Button variant="ghost" onClick={onClose}>
                Cancel
              </Button>
              <Button variant="danger" onClick={onConfirm}>
                Revoke
              </Button>
            </>
          }
        >
          <p className="text-sm text-muted">
            This action is immediate and irreversible.
          </p>
        </Modal>
      </>
    );
  },
};
