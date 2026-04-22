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
    description: [
      "Classify a single job description. Runs NER to detect occupations, skills, and qualifications, then links each entity to ESCO via NEL.",
      "",
      "Request fields:",
      "  • text — raw job ad text (use this OR title+description)",
      "  • title — job title (combined with description if text is omitted)",
      "  • description — job description body",
      "  • options.extract_entities — filter to specific entity types; allowed values: \"occupation\", \"skill\", \"qualification\". Omit to extract all.",
      "  • options.top_k — max ESCO matches per entity (1–50, default 5)",
      "  • options.min_similarity — minimum cosine similarity for a match to appear (0.0–1.0, default 0.0)",
    ].join("\n"),
    requestBody: JSON.stringify(
      {
        text: "Looking for a Senior Data Engineer with Python and SQL experience.",
        options: {
          extract_entities: ["occupation", "skill"],
          top_k: 3,
          min_similarity: 0.5,
        },
      },
      null,
      2
    ),
    responseBody: JSON.stringify(
      {
        classification: {
          entities: [
            {
              entity_type: "occupation",
              surface_form: "Senior Data Engineer",
              span: { start: 15, end: 35 },
              linked_entities: [
                { label: "data engineer", uri: "http://data.europa.eu/esco/occupation/...", score: 0.94 }
              ]
            },
            {
              entity_type: "skill",
              surface_form: "Python",
              span: { start: 41, end: 47 },
              linked_entities: [
                { label: "Python (programming language)", uri: "http://data.europa.eu/esco/skill/...", score: 0.98 }
              ]
            }
          ],
          entity_counts: { occupation: 1, skill: 1 }
        },
        metadata: {
          classifier_version: "1.0.0",
          model_name: "tabiya/roberta-base-job-ner",
          linker_model: "sentence-transformers/...",
          processing_time_ms: 142.3,
          input_text_hash: "a3f5c..."
        }
      },
      null,
      2
    ),
  },
  {
    method: "POST",
    path: "/v1/classify/batch",
    description: [
      "Submit a list of job descriptions for async batch classification. Returns immediately with a batch_id — poll /v1/batch/{batch_id}/status to track progress.",
      "",
      "Request fields:",
      "  • jobs — array of job objects (required, min 1 item)",
      "    – job_id — optional identifier; defaults to \"job_0\", \"job_1\", …",
      "    – text / title / description — same as the single classify endpoint",
      "  • options — same ClassifyOptions as the single endpoint (applied to all jobs)",
    ].join("\n"),
    requestBody: JSON.stringify(
      {
        jobs: [
          { job_id: "job-1", text: "Looking for a nurse with ICU experience." },
          { job_id: "job-2", title: "Software Engineer", description: "Python and Kubernetes required." }
        ],
        options: { top_k: 5, min_similarity: 0.4 }
      },
      null,
      2
    ),
    responseBody: JSON.stringify({ batch_id: "batch_abc123", status: "processing", total: 2 }, null, 2),
  },
  {
    method: "GET",
    path: "/v1/batch/{batch_id}/status",
    description: "Poll the processing status of a batch. status values: \"processing\" | \"completed\" | \"failed\".",
    responseBody: JSON.stringify({ batch_id: "batch_abc123", status: "processing", processed: 1, total: 2 }, null, 2),
  },
  {
    method: "GET",
    path: "/v1/batch/{batch_id}/results",
    description: [
      "Retrieve results for a completed batch. Each result mirrors the single /v1/classify response, augmented with job_id and status.",
      "",
      "Per-job status values: \"completed\" | \"error\"",
      "On error, the result includes an \"error\" string field instead of classification/metadata.",
    ].join("\n"),
    responseBody: JSON.stringify(
      {
        batch_id: "batch_abc123",
        status: "completed",
        total: 2,
        processed: 2,
        results: [
          {
            job_id: "job-1",
            status: "completed",
            classification: { entities: [], entity_counts: {} },
            metadata: { classifier_version: "1.0.0", model_name: "...", linker_model: "...", processing_time_ms: 98.4, input_text_hash: "..." }
          },
          {
            job_id: "job-2",
            status: "error",
            error: "No classifiable text found"
          }
        ]
      },
      null,
      2
    ),
  },
  {
    method: "GET",
    path: "/v1/health",
    description: "Health check. Returns overall status and per-dependency health for the NER and NEL services. Does not require authentication.",
    responseBody: JSON.stringify(
      {
        status: "healthy",
        service: "classify-api",
        dependencies: {
          ner_api: "healthy",
          nel_api: "healthy"
        }
      },
      null,
      2
    ),
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
          <p style={{ color: "#555", fontSize: "0.9rem", whiteSpace: "pre-wrap" }}>{ep.description}</p>

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
