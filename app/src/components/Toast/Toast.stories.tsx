import type { Meta, StoryObj } from "@storybook/react";
import { fn } from "@storybook/test";
import { Button } from "@/components";
import { ToastProvider, useToast } from "@/components";
import type { ToastPlacement } from "@/components";

const meta: Meta = {
  title: "Primitives/Toast",
  parameters: { layout: "fullscreen" },
};
export default meta;

type Story = StoryObj;

function Triggers({ onShow }: { onShow: (tone: string) => void }) {
  const toast = useToast();
  return (
    <div className="flex gap-2 p-10">
      <Button
        onClick={() => {
          onShow("success");
          toast.show({ message: "Configuration saved", tone: "success" });
        }}
      >
        Success
      </Button>
      <Button
        variant="ghost"
        onClick={() => {
          onShow("info");
          toast.show({ message: "Heads up — token expires soon" });
        }}
      >
        Info
      </Button>
      <Button
        variant="danger"
        onClick={() => {
          onShow("error");
          toast.show({ message: "Couldn't reach the classifier", tone: "error" });
        }}
      >
        Error
      </Button>
    </div>
  );
}

export const Default: Story = {
  render: () => {
    const onShow = fn();
    return (
      <ToastProvider>
        <Triggers onShow={onShow} />
      </ToastProvider>
    );
  },
};

const placements: ToastPlacement[] = [
  "top-left",
  "top-center",
  "top-right",
  "bottom-left",
  "bottom-center",
  "bottom-right",
];

export const PlacementShowcase: Story = {
  render: () => {
    const onShow = fn();
    return (
      <div className="grid grid-cols-2 gap-3 p-10">
        {placements.map((placement) => (
          <ToastProvider key={placement} placement={placement}>
            <div className="rounded-md border border-line bg-paper p-4">
              <div className="eyebrow mb-2">{placement}</div>
              <Triggers onShow={onShow} />
            </div>
          </ToastProvider>
        ))}
      </div>
    );
  },
};
