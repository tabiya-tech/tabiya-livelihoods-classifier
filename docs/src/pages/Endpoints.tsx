import React, { useState } from "react";

const H = { fontFamily: "'IBM Plex Mono', monospace", color: "#002147" } as const;

interface Endpoint {
  method: "POST" | "GET";
  path: string;
  description: string;
  requestBody?: string;
  responseBody: string;
}

const ENDPOINTS: Endpoint[] = [
  {
    method: "POST",
    path: "/v1/classify",
    description: "Classify a single job description. Returns extracted entities linked to ESCO.",
    requestBody: JSON.stringify(
      { text: "Looking for a Senior Data Engineer with Python and SQL experience." },
      null,
      2
    ),
    responseBody: JSON.stringify(
      {
        version: "1.0.0",
        entities: [
          {
            entity_type: "Occupation",
            surface_form: "Senior Data Engineer",
            span: { start: 15, end: 35 },
            links: [
              { label: "data engineer", uri: "http://data.europa.eu/esco/occupation/...", score: 0.94 }
            ]
          },
          {
            entity_type: "Skill",
            surface_form: "Python",
            span: { start: 41, end: 47 },
            links: [
              { label: "Python (programming language)", uri: "http://data.europa.eu/esco/skill/...", score: 0.98 }
            ]
          }
        ]
      },
      null,
      2
    ),
  },
  {
    method: "POST",
    path: "/v1/classify/batch",
    description: "Submit a list of job descriptions for batch classification. Returns a batch_id to poll.",
    requestBody: JSON.stringify(
      { items: [{ id: "job-1", text: "..." }, { id: "job-2", text: "..." }] },
      null,
      2
    ),
    responseBody: JSON.stringify({ batch_id: "batch_abc123", status: "queued", total: 2 }, null, 2),
  },
  {
    method: "GET",
    path: "/v1/batch/{batch_id}/status",
    description: "Poll the status of a batch job.",
    responseBody: JSON.stringify({ batch_id: "batch_abc123", status: "processing", processed: 1, total: 2 }, null, 2),
  },
  {
    method: "GET",
    path: "/v1/batch/{batch_id}/results",
    description: "Retrieve results once the batch is complete.",
    responseBody: JSON.stringify(
      {
        batch_id: "batch_abc123",
        status: "complete",
        results: [
          { id: "job-1", entities: [] },
          { id: "job-2", entities: [] }
        ]
      },
      null,
      2
    ),
  },
  {
    method: "GET",
    path: "/v1/health",
    description: "Health check. Does not require authentication.",
    responseBody: JSON.stringify({ status: "ok", version: "1.0.0" }, null, 2),
  },
];

function MethodBadge({ method }: { method: string }) {
  const colors: Record<string, string> = { POST: "#002147", GET: "#26887D" };
  return (
    <span style={{
      display: "inline-block",
      backgroundColor: colors[method] ?? "#555",
      color: "#fff",
      borderRadius: 4,
      padding: "0.1rem 0.5rem",
      fontFamily: "'IBM Plex Mono', monospace",
      fontWeight: 700,
      fontSize: "0.75rem",
      marginRight: "0.75rem",
      verticalAlign: "middle",
    }}>
      {method}
    </span>
  );
}

function EndpointCard({ ep }: { ep: Endpoint }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{
      border: "1px solid #e0ddd9",
      borderRadius: 10,
      marginBottom: "1rem",
      overflow: "hidden",
      backgroundColor: "#fff",
    }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          display: "flex",
          alignItems: "center",
          width: "100%",
          padding: "1rem 1.25rem",
          background: "none",
          border: "none",
          cursor: "pointer",
          textAlign: "left",
        }}
      >
        <MethodBadge method={ep.method} />
        <code style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.9rem", color: "#002147" }}>
          {ep.path}
        </code>
        <span style={{ flex: 1 }} />
        <span style={{ fontSize: "0.8rem", color: "#888" }}>{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div style={{ padding: "0 1.25rem 1.25rem", borderTop: "1px solid #e0ddd9" }}>
          <p style={{ color: "#555", fontSize: "0.9rem" }}>{ep.description}</p>

          {ep.requestBody && (
            <>
              <div style={{ fontSize: "0.8rem", fontWeight: 600, color: "#888", marginBottom: "0.4rem" }}>Request body</div>
              <pre style={{
                backgroundColor: "#002147",
                color: "#00FF91",
                borderRadius: 8,
                padding: "0.75rem 1rem",
                fontSize: "0.82rem",
                overflowX: "auto",
                margin: "0 0 1rem",
              }}>{ep.requestBody}</pre>
            </>
          )}

          <div style={{ fontSize: "0.8rem", fontWeight: 600, color: "#888", marginBottom: "0.4rem" }}>Response</div>
          <pre style={{
            backgroundColor: "#1a2332",
            color: "#F3F1EE",
            borderRadius: 8,
            padding: "0.75rem 1rem",
            fontSize: "0.82rem",
            overflowX: "auto",
            margin: 0,
          }}>{ep.responseBody}</pre>
        </div>
      )}
    </div>
  );
}

export default function EndpointsPage() {
  return (
    <>
      <h1 style={{ ...H, fontSize: "1.75rem", marginTop: 0 }}>API Endpoints</h1>
      <p style={{ color: "#555", marginBottom: "2rem" }}>
        Click an endpoint to expand its request/response schema.
      </p>
      {ENDPOINTS.map((ep) => (
        <EndpointCard key={`${ep.method}-${ep.path}`} ep={ep} />
      ))}
    </>
  );
}
