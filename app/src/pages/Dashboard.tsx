import { Link } from "react-router-dom";
import Layout from "../components/Layout";

export default function Dashboard() {
  return (
    <Layout>
      <h1 style={{ fontFamily: "'IBM Plex Mono', monospace", color: "#002147", marginTop: 0 }}>
        Dashboard
      </h1>

      <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
        {[
          { label: "Configure model →", to: "/configuration" },
          { label: "Manage API keys →", to: "/api-keys" },
        ].map(({ label, to }) => (
          <Link
            key={to}
            to={to}
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
          </Link>
        ))}
      </div>
    </Layout>
  );
}
