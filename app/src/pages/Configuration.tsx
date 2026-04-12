import { useEffect, useState } from "react";
import Layout from "../components/Layout";
import { getConfig, UserConfig } from "../lib/api";

export default function Configuration() {
  const [config, setConfig] = useState<UserConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    getConfig()
      .then(setConfig)
      .catch(() => setError("Failed to load configuration"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <Layout>
      <h1 style={{ fontFamily: "'IBM Plex Mono', monospace", color: "#002147", marginTop: 0 }}>
        Configuration
      </h1>

      <div style={{
        backgroundColor: "#fff",
        border: "1px solid #e0ddd9",
        borderRadius: 12,
        padding: "2rem",
        maxWidth: 560,
        marginBottom: "1.5rem",
      }}>
        <h2 style={{ marginTop: 0, color: "#002147", fontSize: "1rem", fontFamily: "'IBM Plex Mono', monospace" }}>
          Current settings
        </h2>

        {loading && <p style={{ color: "#888", margin: 0 }}>Loading…</p>}
        {error && <p style={{ color: "#c0392b", margin: 0 }}>{error}</p>}
        {!loading && !error && config && (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
            <tbody>
              {Object.entries(config).map(([key, value]) => (
                <tr key={key} style={{ borderBottom: "1px solid #e0ddd9" }}>
                  <td style={{ padding: "0.65rem 0", color: "#555", width: "50%" }}>{key}</td>
                  <td style={{ padding: "0.65rem 0", color: "#002147", fontFamily: "'IBM Plex Mono', monospace", fontWeight: 600 }}>
                    {String(value)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div style={{
        backgroundColor: "#F3F1EE",
        border: "1px solid #e0ddd9",
        borderRadius: 10,
        padding: "1.25rem 1.5rem",
        maxWidth: 560,
        color: "#555",
        fontSize: "0.9rem",
      }}>
        Model and taxonomy configuration is managed by Tabiya. Contact us if you need a custom setup.
      </div>
    </Layout>
  );
}
