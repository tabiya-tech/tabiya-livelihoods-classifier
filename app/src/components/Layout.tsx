import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { signOut } from "firebase/auth";
import { auth } from "../lib/firebase";
import { useAuth } from "../hooks/useAuth";

const NAV_LINKS = [
  { to: "/dashboard", label: "Dashboard" },
  { to: "/configuration", label: "Configuration" },
  { to: "/api-keys", label: "API Keys" },
];

export default function Layout({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const navigate = useNavigate();

  async function handleSignOut() {
    await signOut(auth);
    navigate("/login");
  }

  return (
    <div style={{ display: "flex", minHeight: "100vh", fontFamily: "Inter, sans-serif", backgroundColor: "#F3F1EE" }}>
      {/* Sidebar */}
      <aside style={{
        width: 220,
        backgroundColor: "#002147",
        color: "#F3F1EE",
        display: "flex",
        flexDirection: "column",
        padding: "2rem 1rem",
        gap: "0.5rem",
        flexShrink: 0,
      }}>
        <div style={{
          fontFamily: "'IBM Plex Mono', monospace",
          fontWeight: 700,
          fontSize: "1.1rem",
          color: "#00FF91",
          marginBottom: "2rem",
          letterSpacing: "-0.03em",
        }}>
          Tabiya<br />Classifier
        </div>

        {NAV_LINKS.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            style={({ isActive }) => ({
              padding: "0.6rem 0.75rem",
              borderRadius: 6,
              textDecoration: "none",
              color: isActive ? "#002147" : "#F3F1EE",
              backgroundColor: isActive ? "#00FF91" : "transparent",
              fontWeight: isActive ? 600 : 400,
              fontSize: "0.9rem",
              transition: "background 0.15s",
            })}
          >
            {label}
          </NavLink>
        ))}

        <a
          href={import.meta.env.VITE_DOCS_URL ?? "https://docs.classifier.tabiya.tech"}
          target="_blank"
          rel="noreferrer"
          style={{
            padding: "0.6rem 0.75rem",
            textDecoration: "none",
            color: "#F3F1EE",
            fontSize: "0.9rem",
            opacity: 0.7,
          }}
        >
          Documentation ↗
        </a>

        <div style={{ flexGrow: 1 }} />

        <div style={{ fontSize: "0.8rem", opacity: 0.6, marginBottom: "0.5rem" }}>
          {user?.email}
        </div>
        <button
          onClick={handleSignOut}
          style={{
            background: "transparent",
            border: "1px solid rgba(255,255,255,0.3)",
            color: "#F3F1EE",
            padding: "0.5rem 0.75rem",
            borderRadius: 6,
            cursor: "pointer",
            fontSize: "0.85rem",
          }}
        >
          Sign out
        </button>
      </aside>

      {/* Main content */}
      <main style={{ flex: 1, padding: "2.5rem", overflowY: "auto" }}>
        {children}
      </main>
    </div>
  );
}
