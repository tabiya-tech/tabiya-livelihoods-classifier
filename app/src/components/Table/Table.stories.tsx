import type { Meta, StoryObj } from "@storybook/react";
import { Table } from "./Table";
import { Card } from "@/components";
import { Tag } from "@/components";

const meta: Meta<typeof Table> = {
  title: "Primitives/Table",
  component: Table,
  parameters: { layout: "padded" },
};
export default meta;

type Story = StoryObj<typeof Table>;

const rows = [
  { id: 1, when: "2 min ago", title: "Senior Data Engineer", entities: 21, type: "UI" },
  { id: 2, when: "14 min ago", title: "ICU Registered Nurse", entities: 17, type: "API" },
  { id: 3, when: "1 h ago", title: "Marketing Director, EMEA", entities: 24, type: "UI" },
];

export const Default: Story = {
  render: () => (
    <Card flush>
      <Table>
        <Table.Head>
          <Table.Row>
            <Table.HeaderCell>When</Table.HeaderCell>
            <Table.HeaderCell>Title</Table.HeaderCell>
            <Table.HeaderCell>Source</Table.HeaderCell>
            <Table.HeaderCell>Entities</Table.HeaderCell>
          </Table.Row>
        </Table.Head>
        <Table.Body>
          {rows.map((row) => (
            <Table.Row key={row.id} hover>
              <Table.Cell className="font-mono text-xs text-muted">{row.when}</Table.Cell>
              <Table.Cell className="font-medium">{row.title}</Table.Cell>
              <Table.Cell>
                <Tag size="sm">{row.type}</Tag>
              </Table.Cell>
              <Table.Cell className="font-mono">{row.entities}</Table.Cell>
            </Table.Row>
          ))}
        </Table.Body>
      </Table>
    </Card>
  ),
};
