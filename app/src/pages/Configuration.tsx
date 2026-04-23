import { useEffect, useState } from "react";
import Layout from "../components/Layout";
import {
  getV2Config,
  saveV2Config,
  listNELModels,
  listTaxonomyModels,
  NELModel,
  TaxonomyModel,
  V2UserConfig,
} from "../lib/api";

export default function Configuration() {
  const [nelModels, setNELModels] = useState<NELModel[]>([]);
  const [taxonomyModels, setTaxonomyModels] = useState<TaxonomyModel[]>([]);
  const [selectedNELModel, setSelectedNELModel] = useState("");
  const [selectedTaxonomyModel, setSelectedTaxonomyModel] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [saveError, setSaveError] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    Promise.all([getV2Config(), listNELModels(), listTaxonomyModels()])
      .then(([config, nels, taxonomies]) => {
        setNELModels(nels);
        setTaxonomyModels(taxonomies.filter((t) => t.released));
        setSelectedNELModel(config.nel_model_id);
        setSelectedTaxonomyModel(config.taxonomy_model_id);
      })
      .catch(() => setLoadError("Failed to load configuration"))
      .finally(() => setLoading(false));
  }, []);

  async function handleSave() {
    setSaving(true);
    setSaveError("");
    setSaved(false);
    try {
      await saveV2Config({
        nel_model_id: selectedNELModel,
        taxonomy_model_id: selectedTaxonomyModel,
      });
      setSaved(true);
    } catch {
      setSaveError("Failed to save configuration");
    } finally {
      setSaving(false);
    }
  }

  const selectStyle: React.CSSProperties = {
    width: "100%",
    padding: "0.5rem 0.75rem",
    border: "1px solid #e0ddd9",
    borderRadius: 6,
    fontFamily: "'IBM Plex Mono', monospace",
    fontSize: "0.875rem",
    color: "#002147",
    backgroundColor: "#fff",
    cursor: "pointer",
  };

  const labelStyle: React.CSSProperties = {
    display: "block",
    fontSize: "0.85rem",
    color: "#555",
    marginBottom: "0.4rem",
  };

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
          Model settings
        </h2>

        {loading && <p style={{ color: "#888", margin: 0 }}>Loading…</p>}
        {loadError && <p style={{ color: "#c0392b", margin: 0 }}>{loadError}</p>}

        {!loading && !loadError && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
            <div>
              <label style={labelStyle}>Taxonomy model</label>
              <select
                value={selectedTaxonomyModel}
                onChange={(e) => { setSelectedTaxonomyModel(e.target.value); setSaved(false); }}
                style={selectStyle}
              >
                <option value="">— select —</option>
                {taxonomyModels.map((t) => (
                  <option key={t.id} value={t.id}>{t.name} {t.version}</option>
                ))}
              </select>
            </div>

            <div>
              <label style={labelStyle}>NEL model</label>
              <select
                value={selectedNELModel}
                onChange={(e) => { setSelectedNELModel(e.target.value); setSaved(false); }}
                style={selectStyle}
              >
                <option value="">— select —</option>
                {nelModels.map((m) => (
                  <option key={m.model_id} value={m.model_id}>{m.model_id}</option>
                ))}
              </select>
            </div>

            {saveError && <p style={{ color: "#c0392b", margin: 0, fontSize: "0.875rem" }}>{saveError}</p>}
            {saved && <p style={{ color: "#27ae60", margin: 0, fontSize: "0.875rem" }}>Saved</p>}

            <button
              onClick={handleSave}
              disabled={saving || !selectedNELModel || !selectedTaxonomyModel}
              style={{
                alignSelf: "flex-start",
                padding: "0.5rem 1.25rem",
                backgroundColor: "#002147",
                color: "#fff",
                border: "none",
                borderRadius: 6,
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: "0.875rem",
                cursor: saving ? "wait" : "pointer",
                opacity: (!selectedNELModel || !selectedTaxonomyModel) ? 0.5 : 1,
              }}
            >
              {saving ? "Saving…" : "Save"}
            </button>
          </div>
        )}
      </div>
    </Layout>
  );
}
