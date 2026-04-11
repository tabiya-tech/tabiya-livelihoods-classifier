/**
 * Thin API client for the Tabiya Classifier backend.
 *
 * The classify API URL is set via VITE_API_BASE_URL.
 * All requests send the Firebase ID token as Bearer auth so the backend
 * (or a future BFF) can map the user to their API keys and config.
 */

import { auth } from "./firebase";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:5001";

async function getIdToken(): Promise<string> {
  const user = auth.currentUser;
  if (!user) throw new Error("Not authenticated");
  return user.getIdToken();
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = await getIdToken();
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

// ── User config ───────────────────────────────────────────────────────────

export interface UserConfig {
  ner_type: string;
  nel_type: string;
  ner_model_name: string;
  nel_model_name: string;
  taxonomy_model_id: string;
}

export function getConfig(): Promise<UserConfig> {
  return request<UserConfig>("/v1/user/config");
}

export function saveConfig(config: Partial<UserConfig>): Promise<void> {
  return request<void>("/v1/user/config", {
    method: "PUT",
    body: JSON.stringify(config),
  });
}

// ── API keys ──────────────────────────────────────────────────────────────

export interface ApiKey {
  key_id: string;
  label: string;
  created_at: string;
  last_used_at: string | null;
  revoked: boolean;
  // Suffix shown after creation (last 4 chars of plain key)
  suffix?: string;
}

export function listApiKeys(): Promise<ApiKey[]> {
  return request<ApiKey[]>("/v1/user/api-keys");
}

export function createApiKey(label: string): Promise<{ key: string; meta: ApiKey }> {
  return request<{ key: string; meta: ApiKey }>("/v1/user/api-keys", {
    method: "POST",
    body: JSON.stringify({ label }),
  });
}

export function deleteApiKey(keyId: string): Promise<void> {
  return request<void>(`/v1/user/api-keys/${keyId}`, { method: "DELETE" });
}

// ── Usage ─────────────────────────────────────────────────────────────────

export interface UsagePoint {
  date: string;  // ISO date string
  count: number;
}

export function getUsage(): Promise<UsagePoint[]> {
  return request<UsagePoint[]>("/v1/user/usage");
}
