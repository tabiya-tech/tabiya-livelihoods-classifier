import React, { useState } from "react";

const H = { fontFamily: "'IBM Plex Mono', monospace", color: "#002147" } as const;

const DEFAULT_TEXT =
  "Looking for a Senior Data Engineer with Python and SQL experience in the fintech domain.";

export default function TryIt() {
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("https://api.classifier.tabiya.tech");
  const [text, setText] = useState(DEFAULT_TEXT);
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleRun(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${baseUrl}/v1/classify`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": apiKey,
        },
        body: JSON.stringify({ text }),
      });
      const json = await res.json();
      setResult(JSON.stringify(json, null, 2));
      if (!res.ok) setError(`HTTP ${res.status}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <h1 style={{ ...H, fontSize: "1.75rem", marginTop: 0 }}>Try it now</h1>
      <p style={{ color: "#555", marginBottom: "1.5rem" }}>
        Paste your API key and a job description to send a live classify request.
        Get your API key from the{" "}
        <a href="https://app.classifier.tabiya.tech/api-keys" style={{ color: "#26887D" }}>
          dashboard
        </a>.
      </p>

      <form onSubmit={handleRun} style={{ maxWidth: 620 }}>
        <label style={{ display: "block", fontWeight: 600, color: "#002147", fontSize: "0.85rem", marginBottom: "0.35rem" }}>
          API Key
        </label>
        <input
          type="password"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          required
          placeholder="sk_live_..."
          style={{
            width: "100%",
            padding: "0.6rem 0.75rem",
            border: "1px solid #ccc",
            borderRadius: 6,
            fontSize: "0.9rem",
            boxSizing: "border-box",
            marginBottom: "1rem",
            fontFamily: "'IBM Plex Mono', monospace",
          }}
        />

        <label style={{ display: "block", fontWeight: 600, color: "#002147", fontSize: "0.85rem", marginBottom: "0.35rem" }}>
          API Base URL
        </label>
        <input
          type="url"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value)}
          style={{
            width: "100%",
            padding: "0.6rem 0.75rem",
            border: "1px solid #ccc",
            borderRadius: 6,
            fontSize: "0.9rem",
            boxSizing: "border-box",
            marginBottom: "1rem",
            fontFamily: "'IBM Plex Mono', monospace",
          }}
        />

        <label style={{ display: "block", fontWeight: 600, color: "#002147", fontSize: "0.85rem", marginBottom: "0.35rem" }}>
          Job description text
        </label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={5}
          required
          style={{
            width: "100%",
            padding: "0.6rem 0.75rem",
            border: "1px solid #ccc",
            borderRadius: 6,
            fontSize: "0.9rem",
            boxSizing: "border-box",
            resize: "vertical",
            marginBottom: "1rem",
          }}
        />

        <button
          type="submit"
          disabled={loading}
          style={{
            backgroundColor: loading ? "#888" : "#002147",
            color: "#00FF91",
            border: "none",
            borderRadius: 8,
            padding: "0.75rem 2rem",
            fontFamily: "'IBM Plex Mono', monospace",
            fontWeight: 700,
            fontSize: "0.95rem",
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Running…" : "Run request"}
        </button>
      </form>

      {error && (
        <div style={{ color: "#c0392b", marginTop: "1rem", fontSize: "0.9rem" }}>{error}</div>
      )}

      {result && (
        <div style={{ marginTop: "1.5rem" }}>
          <div style={{ fontSize: "0.85rem", fontWeight: 600, color: "#26887D", marginBottom: "0.5rem" }}>
            Response
          </div>
          <pre style={{
            backgroundColor: "#002147",
            color: "#00FF91",
            borderRadius: 10,
            padding: "1rem 1.25rem",
            fontSize: "0.82rem",
            overflowX: "auto",
            maxHeight: 480,
            overflowY: "auto",
          }}>
            {result}
          </pre>
        </div>
      )}
    </>
  );
}
