import { useEffect, useState } from "react";
import Layout from "../components/Layout";
import { getUsage, UsagePoint } from "../lib/api";

function UsageChart({ data }: { data: UsagePoint[] }) {
  if (!data.length) return <p style={{ color: "#888" }}>No usage data yet.</p>;

  const max = Math.max(...data.map((d) => d.count), 1);
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 80 }}>
      {data.map((d) => (
        <div
          key={d.date}
          title={`${d.date}: ${d.count} requests`}
          style={{
            flex: 1,
            height: `${(d.count / max) * 100}%`,
            backgroundColor: "#00FF91",
            minHeight: 2,
            borderRadius: "2px 2px 0 0",
            cursor: "default",
          }}
        />
      ))}
    </div>
  );
}

export default function Dashboard() {
  const [usage, setUsage] = useState<UsagePoint[]>([]);
  const [totalToday, setTotalToday] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    getUsage()
      .then((data: UsagePoint[]) => {
        setUsage(data);
        const today = new Date().toISOString().slice(0, 10);
        const todayPoint = data.find((d: UsagePoint) => d.date === today);
        setTotalToday(todayPoint?.count ?? 0);
      })
      .catch(() => setError("Failed to load usage data"))
      .finally(() => setLoading(false));
  }, []);

  const total30d = usage.reduce((s, d) => s + d.count, 0);

  return (
    <Layout>
      <h1 style={{ fontFamily: "'IBM Plex Mono', monospace", color: "#002147", marginTop: 0 }}>
        Dashboard
      </h1>

      {/* Stat cards */}
      <div style={{ display: "flex", gap: "1rem", marginBottom: "2rem", flexWrap: "wrap" }}>
        {[
          { label: "Requests today", value: loading || error ? "—" : totalToday },
          { label: "Requests (30 d)", value: loading || error ? "—" : total30d },
        ].map(({ label, value }) => (
          <div
            key={label}
            style={{
              flex: "1 1 160px",
              backgroundColor: "#fff",
              border: "1px solid #e0ddd9",
              borderRadius: 10,
              padding: "1.25rem 1.5rem",
            }}
          >
            <div style={{ fontSize: "0.8rem", color: "#888", marginBottom: "0.5rem" }}>{label}</div>
            <div style={{
              fontFamily: "'IBM Plex Mono', monospace",
              fontWeight: 700,
              fontSize: "2rem",
              color: "#002147",
            }}>
              {value}
            </div>
          </div>
        ))}
      </div>

      {/* Usage chart */}
      <div style={{
        backgroundColor: "#fff",
        border: "1px solid #e0ddd9",
        borderRadius: 10,
        padding: "1.5rem",
        marginBottom: "2rem",
      }}>
        <h3 style={{ margin: "0 0 1rem", color: "#002147", fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.95rem" }}>
          Requests — last 30 days
        </h3>
        {loading ? (
          <p style={{ color: "#888" }}>Loading…</p>
        ) : error ? (
          <p style={{ color: "#c0392b", margin: 0 }}>{error}</p>
        ) : (
          <UsageChart data={usage} />
        )}
      </div>

      {/* Quick links */}
      <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
        {[
          { label: "Manage API keys →", to: "/api-keys" },
          { label: "Configure model →", to: "/configuration" },
        ].map(({ label, to }) => (
          <a
            key={to}
            href={to}
            style={{
              padding: "0.75rem 1.25rem",
              backgroundColor: "#EEFF41",
              color: "#002147",
              borderRadius: 8,
              textDecoration: "none",
              fontWeight: 600,
              fontSize: "0.9rem",
            }}
          >
            {label}
          </a>
        ))}
      </div>
    </Layout>
  );
}
