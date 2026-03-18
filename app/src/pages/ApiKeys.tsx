import React, { useEffect, useState } from "react";
import Layout from "../components/Layout";
import { listApiKeys, createApiKey, deleteApiKey, ApiKey } from "../lib/api";

const MAX_KEYS = 5;

function KeyRow({ k, onDelete }: { k: ApiKey; onDelete: () => void }) {
  const [confirming, setConfirming] = useState(false);

  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "0.85rem 0",
      borderBottom: "1px solid #e0ddd9",
      flexWrap: "wrap",
      gap: "0.5rem",
    }}>
      <div>
        <div style={{ fontWeight: 600, color: "#002147", fontSize: "0.9rem" }}>{k.label}</div>
        <div style={{ fontSize: "0.8rem", color: "#888", fontFamily: "'IBM Plex Mono', monospace" }}>
          {k.key_id}
        </div>
        <div style={{ fontSize: "0.75rem", color: "#aaa" }}>
          Created {new Date(k.created_at).toLocaleDateString()}
          {k.last_used_at && ` · Last used ${new Date(k.last_used_at).toLocaleDateString()}`}
        </div>
      </div>

      {!confirming ? (
        <button
          onClick={() => setConfirming(true)}
          style={{
            background: "none",
            border: "1px solid #e0ddd9",
            borderRadius: 6,
            padding: "0.4rem 0.75rem",
            cursor: "pointer",
            fontSize: "0.8rem",
            color: "#c0392b",
          }}
        >
          Revoke
        </button>
      ) : (
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button
            onClick={onDelete}
            style={{
              backgroundColor: "#c0392b",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              padding: "0.4rem 0.75rem",
              cursor: "pointer",
              fontSize: "0.8rem",
            }}
          >
            Confirm revoke
          </button>
          <button
            onClick={() => setConfirming(false)}
            style={{
              background: "none",
              border: "1px solid #ccc",
              borderRadius: 6,
              padding: "0.4rem 0.75rem",
              cursor: "pointer",
              fontSize: "0.8rem",
            }}
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}

export default function ApiKeys() {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [label, setLabel] = useState("");
  const [newKey, setNewKey] = useState<string | null>(null);
  const [error, setError] = useState("");

  function load() {
    listApiKeys()
      .then(setKeys)
      .catch(() => setError("Failed to load API keys"))
      .finally(() => setLoading(false));
  }

  useEffect(() => { load(); }, []);

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!label.trim()) return;
    setCreating(true);
    setError("");
    try {
      const { key, meta } = await createApiKey(label.trim());
      setNewKey(key);
      setLabel("");
      setKeys((prev) => [...prev, meta]);
    } catch {
      setError("Failed to create key");
    } finally {
      setCreating(false);
    }
  }

  async function handleDelete(keyId: string) {
    try {
      await deleteApiKey(keyId);
      setKeys((prev) => prev.filter((k) => k.key_id !== keyId));
    } catch {
      setError("Failed to revoke key");
    }
  }

  return (
    <Layout>
      <h1 style={{ fontFamily: "'IBM Plex Mono', monospace", color: "#002147", marginTop: 0 }}>
        API Keys
      </h1>
      <p style={{ color: "#555", marginBottom: "2rem", maxWidth: 580 }}>
        API keys authenticate your requests to the Classify API. Keys are shown once — store them safely.
        Maximum {MAX_KEYS} keys per account.
      </p>

      {/* New key banner */}
      {newKey && (
        <div style={{
          backgroundColor: "#002147",
          color: "#00FF91",
          borderRadius: 10,
          padding: "1.25rem 1.5rem",
          marginBottom: "1.5rem",
          maxWidth: 600,
        }}>
          <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>
            New API key created — copy it now, it won't be shown again.
          </div>
          <code style={{
            display: "block",
            backgroundColor: "rgba(0,0,0,0.3)",
            padding: "0.6rem 0.75rem",
            borderRadius: 6,
            fontSize: "0.85rem",
            wordBreak: "break-all",
          }}>
            {newKey}
          </code>
          <button
            onClick={() => { navigator.clipboard.writeText(newKey); }}
            style={{
              marginTop: "0.75rem",
              background: "none",
              border: "1px solid #00FF91",
              color: "#00FF91",
              borderRadius: 6,
              padding: "0.4rem 0.75rem",
              cursor: "pointer",
              fontSize: "0.8rem",
            }}
          >
            Copy to clipboard
          </button>
          <button
            onClick={() => setNewKey(null)}
            style={{
              marginTop: "0.75rem",
              marginLeft: "0.5rem",
              background: "none",
              border: "none",
              color: "rgba(255,255,255,0.5)",
              cursor: "pointer",
              fontSize: "0.8rem",
            }}
          >
            Dismiss
          </button>
        </div>
      )}

      {error && (
        <div style={{ color: "#c0392b", marginBottom: "1rem", fontSize: "0.9rem" }}>{error}</div>
      )}

      {/* Create form */}
      {keys.length < MAX_KEYS && (
        <form
          onSubmit={handleCreate}
          style={{
            display: "flex",
            gap: "0.75rem",
            marginBottom: "2rem",
            maxWidth: 500,
            flexWrap: "wrap",
          }}
        >
          <input
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="Key label (e.g. Production)"
            required
            style={{
              flex: 1,
              padding: "0.6rem 0.75rem",
              border: "1px solid #ccc",
              borderRadius: 6,
              fontSize: "0.9rem",
              minWidth: 180,
            }}
          />
          <button
            type="submit"
            disabled={creating}
            style={{
              backgroundColor: "#002147",
              color: "#00FF91",
              border: "none",
              borderRadius: 6,
              padding: "0.6rem 1.25rem",
              fontWeight: 700,
              cursor: creating ? "not-allowed" : "pointer",
              fontFamily: "'IBM Plex Mono', monospace",
              fontSize: "0.85rem",
            }}
          >
            {creating ? "Creating…" : "Create key"}
          </button>
        </form>
      )}

      {/* Keys list */}
      <div style={{
        backgroundColor: "#fff",
        border: "1px solid #e0ddd9",
        borderRadius: 12,
        padding: "0.5rem 1.5rem",
        maxWidth: 600,
      }}>
        {loading ? (
          <p style={{ color: "#888", padding: "1rem 0" }}>Loading…</p>
        ) : keys.length === 0 ? (
          <p style={{ color: "#888", padding: "1rem 0" }}>No API keys yet. Create one above.</p>
        ) : (
          keys.map((k) => (
            <KeyRow key={k.key_id} k={k} onDelete={() => handleDelete(k.key_id)} />
          ))
        )}
      </div>

      {keys.length >= MAX_KEYS && (
        <p style={{ color: "#888", marginTop: "1rem", fontSize: "0.85rem" }}>
          Maximum of {MAX_KEYS} keys reached. Revoke an existing key to create a new one.
        </p>
      )}
    </Layout>
  );
}
