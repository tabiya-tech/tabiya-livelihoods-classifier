/**
 * Configuration wizard — Sentry-style multi-step flow.
 * Lets users choose their NER model, NEL type, and taxonomy.
 */

import { useEffect, useState } from "react";
import Layout from "../components/Layout";
import { getConfig, saveConfig, UserConfig } from "../lib/api";

// ── Options ───────────────────────────────────────────────────────────────

const NER_OPTIONS = [
  {
    value: "SELF_HOSTED_LLM",
    label: "Tabiya Hosted (RoBERTa)",
    description: "High accuracy. Pre-trained on job descriptions. No setup required.",
    tradeoffs: "Best accuracy · Managed by Tabiya · No data customisation",
  },
  {
    value: "PARTNER_FINE_TUNED",
    label: "Partner Fine-tuned Model",
    description: "Customised model trained on your specific job data.",
    tradeoffs: "Highest accuracy for your domain · Requires training data · Setup required",
  },
];

const NEL_OPTIONS = [
  {
    value: "generic",
    label: "Generic ESCO",
    description: "Links entities to the standard ESCO taxonomy.",
    tradeoffs: "Works out of the box · Broad ESCO coverage",
  },
  {
    value: "partner_specific",
    label: "Partner-specific Taxonomy",
    description: "Links entities to your custom taxonomy or ESCO subset.",
    tradeoffs: "Tailored to your context · Requires custom taxonomy data",
  },
];

const TAXONOMY_OPTIONS = [
  { value: "ui1u2jn", label: "ESCO v1.1.1" },
  { value: "esco_v1_2", label: "ESCO v1.2" },
];

// ── Sub-components ────────────────────────────────────────────────────────

function OptionCard({
  option,
  selected,
  onSelect,
}: {
  option: { value: string; label: string; description: string; tradeoffs: string };
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      style={{
        display: "block",
        width: "100%",
        textAlign: "left",
        padding: "1rem 1.25rem",
        border: `2px solid ${selected ? "#002147" : "#e0ddd9"}`,
        borderRadius: 10,
        backgroundColor: selected ? "#002147" : "#fff",
        color: selected ? "#F3F1EE" : "#002147",
        cursor: "pointer",
        marginBottom: "0.75rem",
        transition: "all 0.15s",
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: "0.25rem" }}>{option.label}</div>
      <div style={{ fontSize: "0.85rem", opacity: 0.8, marginBottom: "0.4rem" }}>{option.description}</div>
      <div style={{
        fontSize: "0.75rem",
        color: selected ? "#00FF91" : "#26887D",
        fontFamily: "'IBM Plex Mono', monospace",
      }}>
        {option.tradeoffs}
      </div>
    </button>
  );
}

function Step({
  number,
  title,
  active,
}: {
  number: number;
  title: string;
  active: boolean;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "1.5rem" }}>
      <div style={{
        width: 28,
        height: 28,
        borderRadius: "50%",
        backgroundColor: active ? "#002147" : "#e0ddd9",
        color: active ? "#00FF91" : "#888",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "'IBM Plex Mono', monospace",
        fontWeight: 700,
        fontSize: "0.85rem",
        flexShrink: 0,
      }}>
        {number}
      </div>
      <span style={{ fontWeight: active ? 700 : 400, color: active ? "#002147" : "#888", fontSize: "0.95rem" }}>
        {title}
      </span>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────────────

export default function Configuration() {
  const [config, setConfig] = useState<Partial<UserConfig>>({
    ner_type: "SELF_HOSTED_LLM",
    nel_type: "generic",
    taxonomy_model_id: "ui1u2jn",
  });
  const [step, setStep] = useState(1);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [supportOpen, setSupportOpen] = useState(false);

  useEffect(() => {
    getConfig()
      .then((c: UserConfig) => setConfig(c))
      .catch(() => {/* no config yet — use defaults */});
  }, []);

  async function handleSave() {
    setSaving(true);
    try {
      await saveConfig(config);
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } finally {
      setSaving(false);
    }
  }

  const steps = ["NER Model", "NEL Type", "Taxonomy", "Review"];

  return (
    <Layout>
      <h1 style={{ fontFamily: "'IBM Plex Mono', monospace", color: "#002147", marginTop: 0 }}>
        Configuration
      </h1>
      <p style={{ color: "#555", marginBottom: "2rem", maxWidth: 580 }}>
        Choose the models and taxonomy for your classification requests. You can change this at any time.
      </p>

      {/* Step indicators */}
      <div style={{ display: "flex", gap: "2rem", marginBottom: "2rem", flexWrap: "wrap" }}>
        {steps.map((s, i) => (
          <Step key={s} number={i + 1} title={s} active={step === i + 1} />
        ))}
      </div>

      <div style={{
        backgroundColor: "#fff",
        border: "1px solid #e0ddd9",
        borderRadius: 12,
        padding: "2rem",
        maxWidth: 580,
      }}>
        {/* Step 1 — NER */}
        {step === 1 && (
          <>
            <h2 style={{ marginTop: 0, color: "#002147", fontSize: "1.1rem" }}>NER Model</h2>
            <p style={{ color: "#555", fontSize: "0.9rem" }}>
              The Named Entity Recognition model extracts skills, occupations, and qualifications from text.
            </p>
            {NER_OPTIONS.map((o) => (
              <OptionCard
                key={o.value}
                option={o}
                selected={config.ner_type === o.value}
                onSelect={() => setConfig((c: Partial<UserConfig>) => ({ ...c, ner_type: o.value }))}
              />
            ))}
            <div style={{ marginTop: "0.5rem" }}>
              <button
                style={{ background: "none", border: "none", color: "#26887D", cursor: "pointer", fontSize: "0.85rem", padding: 0 }}
                onClick={() => setSupportOpen(true)}
              >
                Need a custom model? Request support →
              </button>
            </div>
          </>
        )}

        {/* Step 2 — NEL */}
        {step === 2 && (
          <>
            <h2 style={{ marginTop: 0, color: "#002147", fontSize: "1.1rem" }}>NEL Type</h2>
            <p style={{ color: "#555", fontSize: "0.9rem" }}>
              The Named Entity Linking model maps extracted entities to a structured taxonomy.
            </p>
            {NEL_OPTIONS.map((o) => (
              <OptionCard
                key={o.value}
                option={o}
                selected={config.nel_type === o.value}
                onSelect={() => setConfig((c: Partial<UserConfig>) => ({ ...c, nel_type: o.value }))}
              />
            ))}
          </>
        )}

        {/* Step 3 — Taxonomy */}
        {step === 3 && (
          <>
            <h2 style={{ marginTop: 0, color: "#002147", fontSize: "1.1rem" }}>Taxonomy Version</h2>
            <p style={{ color: "#555", fontSize: "0.9rem" }}>
              Choose which version of the ESCO taxonomy to link entities against.
            </p>
            {TAXONOMY_OPTIONS.map((o) => (
              <button
                key={o.value}
                onClick={() => setConfig((c: Partial<UserConfig>) => ({ ...c, taxonomy_model_id: o.value }))}
                style={{
                  display: "block",
                  width: "100%",
                  textAlign: "left",
                  padding: "0.9rem 1.25rem",
                  border: `2px solid ${config.taxonomy_model_id === o.value ? "#002147" : "#e0ddd9"}`,
                  borderRadius: 10,
                  backgroundColor: config.taxonomy_model_id === o.value ? "#002147" : "#fff",
                  color: config.taxonomy_model_id === o.value ? "#F3F1EE" : "#002147",
                  cursor: "pointer",
                  marginBottom: "0.75rem",
                  fontWeight: 600,
                }}
              >
                {o.label}
              </button>
            ))}
          </>
        )}

        {/* Step 4 — Review */}
        {step === 4 && (
          <>
            <h2 style={{ marginTop: 0, color: "#002147", fontSize: "1.1rem" }}>Review & Save</h2>
            {[
              { label: "NER Model", value: NER_OPTIONS.find((o) => o.value === config.ner_type)?.label ?? config.ner_type },
              { label: "NEL Type", value: NEL_OPTIONS.find((o) => o.value === config.nel_type)?.label ?? config.nel_type },
              { label: "Taxonomy", value: TAXONOMY_OPTIONS.find((o) => o.value === config.taxonomy_model_id)?.label ?? config.taxonomy_model_id },
            ].map(({ label, value }) => (
              <div
                key={label}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "0.75rem 0",
                  borderBottom: "1px solid #e0ddd9",
                  fontSize: "0.9rem",
                }}
              >
                <span style={{ color: "#555" }}>{label}</span>
                <span style={{ fontWeight: 600, color: "#002147" }}>{value}</span>
              </div>
            ))}

            <button
              onClick={handleSave}
              disabled={saving}
              style={{
                marginTop: "1.5rem",
                width: "100%",
                padding: "0.85rem",
                backgroundColor: saving ? "#888" : "#002147",
                color: "#00FF91",
                border: "none",
                borderRadius: 8,
                fontFamily: "'IBM Plex Mono', monospace",
                fontWeight: 700,
                fontSize: "1rem",
                cursor: saving ? "not-allowed" : "pointer",
              }}
            >
              {saving ? "Saving…" : saved ? "Saved!" : "Save configuration"}
            </button>
          </>
        )}

        {/* Nav buttons */}
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: "1.5rem" }}>
          <button
            onClick={() => setStep((s) => Math.max(1, s - 1))}
            disabled={step === 1}
            style={{
              background: "none",
              border: "1px solid #ccc",
              borderRadius: 6,
              padding: "0.5rem 1rem",
              cursor: step === 1 ? "not-allowed" : "pointer",
              opacity: step === 1 ? 0.4 : 1,
            }}
          >
            Back
          </button>
          {step < 4 && (
            <button
              onClick={() => setStep((s) => Math.min(4, s + 1))}
              style={{
                backgroundColor: "#EEFF41",
                color: "#002147",
                border: "none",
                borderRadius: 6,
                padding: "0.5rem 1rem",
                fontWeight: 600,
                cursor: "pointer",
              }}
            >
              Next
            </button>
          )}
        </div>
      </div>

      {/* Support request modal */}
      {supportOpen && (
        <div style={{
          position: "fixed",
          inset: 0,
          backgroundColor: "rgba(0,33,71,0.5)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 1000,
        }}>
          <div style={{
            backgroundColor: "#F3F1EE",
            borderRadius: 12,
            padding: "2rem",
            width: "100%",
            maxWidth: 440,
          }}>
            <h2 style={{ marginTop: 0, color: "#002147", fontFamily: "'IBM Plex Mono', monospace" }}>
              Request Support
            </h2>
            <p style={{ color: "#555", fontSize: "0.9rem" }}>
              Tell us about your use case and we'll get back to you about a custom model or taxonomy.
            </p>
            <textarea
              rows={5}
              placeholder="Describe your data, language, domain, and scale…"
              style={{
                width: "100%",
                borderRadius: 6,
                border: "1px solid #ccc",
                padding: "0.75rem",
                fontSize: "0.9rem",
                boxSizing: "border-box",
                resize: "vertical",
              }}
            />
            <div style={{ display: "flex", gap: "0.75rem", marginTop: "1rem" }}>
              <button
                onClick={() => setSupportOpen(false)}
                style={{
                  flex: 1,
                  padding: "0.75rem",
                  background: "none",
                  border: "1px solid #ccc",
                  borderRadius: 6,
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  // In production: POST to /v1/user/support-request
                  alert("Request sent! We'll be in touch.");
                  setSupportOpen(false);
                }}
                style={{
                  flex: 1,
                  padding: "0.75rem",
                  backgroundColor: "#002147",
                  color: "#00FF91",
                  border: "none",
                  borderRadius: 6,
                  fontWeight: 700,
                  cursor: "pointer",
                }}
              >
                Send request
              </button>
            </div>
          </div>
        </div>
      )}
    </Layout>
  );
}
