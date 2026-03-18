import React, { useState } from "react";
import { useNavigate, Navigate } from "react-router-dom";
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  GoogleAuthProvider,
  signInWithPopup,
} from "firebase/auth";
import { auth } from "../lib/firebase";
import { useAuth } from "../hooks/useAuth";

const S = {
  page: {
    minHeight: "100vh",
    backgroundColor: "#002147",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "Inter, sans-serif",
  } as React.CSSProperties,
  card: {
    backgroundColor: "#F3F1EE",
    borderRadius: 12,
    padding: "2.5rem",
    width: "100%",
    maxWidth: 420,
  } as React.CSSProperties,
  logo: {
    fontFamily: "'IBM Plex Mono', monospace",
    fontWeight: 700,
    fontSize: "1.5rem",
    color: "#002147",
    marginBottom: "0.25rem",
  } as React.CSSProperties,
  subtitle: { color: "#26887D", fontSize: "0.9rem", marginBottom: "2rem" } as React.CSSProperties,
  label: { fontSize: "0.85rem", fontWeight: 600, color: "#002147", display: "block", marginBottom: "0.35rem" } as React.CSSProperties,
  input: {
    width: "100%",
    padding: "0.6rem 0.75rem",
    border: "1px solid #ccc",
    borderRadius: 6,
    fontSize: "0.95rem",
    marginBottom: "1rem",
    boxSizing: "border-box" as const,
  } as React.CSSProperties,
  primaryBtn: {
    width: "100%",
    padding: "0.75rem",
    backgroundColor: "#002147",
    color: "#00FF91",
    border: "none",
    borderRadius: 6,
    fontFamily: "'IBM Plex Mono', monospace",
    fontWeight: 700,
    fontSize: "0.95rem",
    cursor: "pointer",
    marginBottom: "0.75rem",
  } as React.CSSProperties,
  googleBtn: {
    width: "100%",
    padding: "0.75rem",
    backgroundColor: "#fff",
    color: "#002147",
    border: "1px solid #ccc",
    borderRadius: 6,
    fontSize: "0.9rem",
    cursor: "pointer",
    marginBottom: "1.5rem",
  } as React.CSSProperties,
  toggle: { fontSize: "0.85rem", color: "#26887D", cursor: "pointer", background: "none", border: "none", padding: 0 } as React.CSSProperties,
  error: { color: "#c0392b", fontSize: "0.85rem", marginBottom: "1rem" } as React.CSSProperties,
};

export default function Login() {
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  if (!loading && user) return <Navigate to="/dashboard" replace />;

  async function handleEmail(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    try {
      if (isSignUp) {
        await createUserWithEmailAndPassword(auth, email, password);
      } else {
        await signInWithEmailAndPassword(auth, email, password);
      }
      navigate("/dashboard");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    }
  }

  async function handleGoogle() {
    setError("");
    try {
      await signInWithPopup(auth, new GoogleAuthProvider());
      navigate("/dashboard");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Google sign-in failed");
    }
  }

  return (
    <div style={S.page}>
      <div style={S.card}>
        <div style={S.logo}>Tabiya Classifier</div>
        <div style={S.subtitle}>
          {isSignUp ? "Create your account" : "Sign in to your dashboard"}
        </div>

        {error && <div style={S.error}>{error}</div>}

        <form onSubmit={handleEmail}>
          <label style={S.label}>Email</label>
          <input
            style={S.input}
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            autoComplete="email"
          />
          <label style={S.label}>Password</label>
          <input
            style={S.input}
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            autoComplete={isSignUp ? "new-password" : "current-password"}
          />
          <button type="submit" style={S.primaryBtn}>
            {isSignUp ? "Create account" : "Sign in"}
          </button>
        </form>

        <button style={S.googleBtn} onClick={handleGoogle}>
          Continue with Google
        </button>

        <button style={S.toggle} onClick={() => setIsSignUp((s) => !s)}>
          {isSignUp ? "Already have an account? Sign in" : "No account? Create one"}
        </button>
      </div>
    </div>
  );
}
