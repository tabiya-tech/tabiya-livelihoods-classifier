/**
 * Stores and retrieves the user's active GCP API key from localStorage.
 *
 * The plaintext key is only returned once (at creation time). We persist it
 * here so the fetcher can attach it to every request without requiring a
 * Firebase token. Each browser keeps its own key independently.
 */

const STORAGE_KEY = "tabiya:activeApiKey";

export interface StoredApiKey {
  key_id: string;
  key_string: string;
  label: string;
}

export function getActiveApiKey(): StoredApiKey | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as StoredApiKey) : null;
  } catch {
    return null;
  }
}

export function setActiveApiKey(key: StoredApiKey): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(key));
}

export function clearActiveApiKey(): void {
  localStorage.removeItem(STORAGE_KEY);
}
