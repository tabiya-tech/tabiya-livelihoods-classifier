import type { Meta, StoryObj } from "@storybook/react";
import type { PipelineValidationIssue } from "@/lib/api";
import { PipelineSaveBar } from "./PipelineSaveBar";

const fixtureSlotMismatchIssue: PipelineValidationIssue = {
  code: "slot_mismatch",
  message: "Output slot RawText does not match input slot Entities",
  stage_index: 1,
  plugin_id: "tabiya.nel.v1",
};

const meta: Meta<typeof PipelineSaveBar> = {
  title: "Features/Pipelines/PipelineSaveBar",
  component: PipelineSaveBar,
  parameters: { layout: "fullscreen" },
  args: {
    isReadonly: false,
    issues: [],
    isValidating: false,
    isDirty: false,
    isSaving: false,
    onSave: () => {},
    onCancel: () => {},
  },
};
export default meta;

type Story = StoryObj<typeof PipelineSaveBar>;

export const Validating: Story = {
  args: {
    isValidating: true,
    isDirty: true,
  },
};

export const Valid: Story = {
  args: {
    issues: [],
    isValidating: false,
    isDirty: true,
  },
};

export const WithIssues: Story = {
  args: {
    issues: [fixtureSlotMismatchIssue],
    isValidating: false,
    isDirty: true,
  },
};

export const Saving: Story = {
  args: {
    isSaving: true,
    isDirty: true,
  },
};

export const Readonly: Story = {
  args: {
    isReadonly: true,
  },
};
