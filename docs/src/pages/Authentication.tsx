import React from "react";

const H = { fontFamily: "'IBM Plex Mono', monospace", color: "#002147" } as const;

function Code({ children }: { children: string }) {
  return (
    <code style={{
      display: "block",
      backgroundColor: "#002147",
      color: "#00FF91",
      padding: "0.75rem 1rem",
      borderRadius: 8,
      fontFamily: "'IBM Plex Mono', monospace",
      fontSize: "0.85rem",
      whiteSpace: "pre",
      overflowX: "auto",
      marginBottom: "1.5rem",
    }}>
      {children}
    </code>
  );
}

export default function Authentication() {
  return (
    <>
      <h1 style={{ ...H, fontSize: "1.75rem", marginTop: 0 }}>Authentication</h1>
      <p style={{ color: "#555", lineHeight: 1.7 }}>
        All API requests require an API key passed in the <strong>x-api-key</strong> header.
        Create and manage your keys in the{" "}
        <a href="https://app.classifier.tabiya.tech/api-keys" style={{ color: "#26887D" }}>
          dashboard
        </a>.
      </p>

      <h2 style={{ ...H, fontSize: "1.1rem" }}>Example</h2>
      <Code>{`curl https://api.classifier.tabiya.tech/v1/classify \\
  -H "x-api-key: YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Looking for a Senior Data Engineer with Python and SQL experience."}'`}</Code>

      <h2 style={{ ...H, fontSize: "1.1rem" }}>Error responses</h2>
      <table style={{ borderCollapse: "collapse", width: "100%", fontSize: "0.9rem" }}>
        <thead>
          <tr style={{ borderBottom: "2px solid #e0ddd9" }}>
            <th style={{ textAlign: "left", padding: "0.5rem" }}>Status</th>
            <th style={{ textAlign: "left", padding: "0.5rem" }}>Meaning</th>
          </tr>
        </thead>
        <tbody>
          {[
            [401, "Missing or invalid x-api-key header"],
            [403, "Key is revoked or rate limit exceeded"],
            [429, "Too many requests — back off and retry"],
          ].map(([status, desc]) => (
            <tr key={status} style={{ borderBottom: "1px solid #e0ddd9" }}>
              <td style={{ padding: "0.5rem", fontFamily: "'IBM Plex Mono', monospace" }}>{status}</td>
              <td style={{ padding: "0.5rem", color: "#555" }}>{desc}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
