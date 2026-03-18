import React from "react";

const H = { fontFamily: "'IBM Plex Mono', monospace", color: "#002147" } as const;

export default function Overview() {
  return (
    <>
      <h1 style={{ ...H, fontSize: "2rem", marginTop: 0 }}>Tabiya Classifier API</h1>
      <p style={{ color: "#555", lineHeight: 1.7, maxWidth: 640 }}>
        The Tabiya Livelihoods Classifier extracts occupations, skills, qualifications,
        experience, and domain entities from job descriptions and links them to the
        <a href="https://esco.ec.europa.eu" target="_blank" rel="noreferrer" style={{ color: "#26887D" }}> ESCO taxonomy</a>.
      </p>

      <h2 style={{ ...H, fontSize: "1.2rem" }}>Base URL</h2>
      <code style={{
        display: "block",
        backgroundColor: "#002147",
        color: "#00FF91",
        padding: "0.75rem 1rem",
        borderRadius: 8,
        fontFamily: "'IBM Plex Mono', monospace",
        fontSize: "0.9rem",
        marginBottom: "1.5rem",
      }}>
        https://api.classifier.tabiya.tech
      </code>

      <h2 style={{ ...H, fontSize: "1.2rem" }}>Pipeline</h2>
      <ol style={{ color: "#555", lineHeight: 2 }}>
        <li><strong>NER</strong> — identifies entity spans in the text</li>
        <li><strong>NEL</strong> — links each span to the ESCO taxonomy</li>
        <li><strong>Classify</strong> — orchestrates both and returns a unified result</li>
      </ol>

      <h2 style={{ ...H, fontSize: "1.2rem" }}>Versioning</h2>
      <p style={{ color: "#555" }}>All endpoints are versioned under <code>/v1/</code>.</p>
    </>
  );
}
