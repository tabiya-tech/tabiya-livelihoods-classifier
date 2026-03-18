import React from "react";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import Overview from "./pages/Overview";
import Authentication from "./pages/Authentication";
import EndpointsPage from "./pages/Endpoints";
import TryIt from "./pages/TryIt";

const FONT = { fontFamily: "Inter, sans-serif" };
const MONO = { fontFamily: "'IBM Plex Mono', monospace" };

const NAV = [
  { to: "/", label: "Overview" },
  { to: "/authentication", label: "Authentication" },
  { to: "/endpoints", label: "Endpoints" },
  { to: "/try-it", label: "Try it now" },
];

function Sidebar() {
  return (
    <aside style={{
      width: 220,
      backgroundColor: "#002147",
      minHeight: "100vh",
      padding: "2rem 1rem",
      flexShrink: 0,
    }}>
      <div style={{ ...MONO, fontWeight: 700, fontSize: "1.1rem", color: "#00FF91", marginBottom: "2rem" }}>
        Tabiya<br />Classifier API
      </div>
      {NAV.map(({ to, label }) => (
        <NavLink
          key={to}
          to={to}
          end={to === "/"}
          style={({ isActive }) => ({
            display: "block",
            padding: "0.6rem 0.75rem",
            borderRadius: 6,
            textDecoration: "none",
            color: isActive ? "#002147" : "#F3F1EE",
            backgroundColor: isActive ? "#00FF91" : "transparent",
            fontWeight: isActive ? 600 : 400,
            fontSize: "0.9rem",
            marginBottom: "0.25rem",
          })}
        >
          {label}
        </NavLink>
      ))}
    </aside>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ display: "flex", ...FONT, backgroundColor: "#F3F1EE", minHeight: "100vh" }}>
        <Sidebar />
        <main style={{ flex: 1, padding: "3rem", maxWidth: 760, overflowY: "auto" }}>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/authentication" element={<Authentication />} />
            <Route path="/endpoints" element={<EndpointsPage />} />
            <Route path="/try-it" element={<TryIt />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
