# MongoDB Atlas Schema

Database: `tabiya-classifier` (configurable via `MONGODB_DB_NAME`)

---

## Collections

### `users`

Created on first Firebase login via the dashboard app.

```json
{
  "_id": ObjectId,
  "user_id": "usr_abc123",          // UUID v4, internal identifier
  "email": "user@example.com",
  "firebase_uid": "firebase_uid_...",
  "created_at": ISODate,
  "updated_at": ISODate
}
```

Indexes:
- `{ firebase_uid: 1 }` — unique, used on login
- `{ user_id: 1 }` — unique, used by api_keys join

---

### `api_keys`

One document per API key. The plain key is shown to the user exactly once;
only the SHA-256 hash is stored.

```json
{
  "_id": ObjectId,
  "key_id": "key_abc123",            // public identifier (not secret)
  "key_hash": "sha256:abc...",       // SHA-256(plain_key), hex-encoded
  "user_id": "usr_abc123",           // → users.user_id
  "label": "My production key",
  "created_at": ISODate,
  "last_used_at": ISODate,           // updated on each authenticated request
  "rate_limit": 1000,                // requests per day (null = default)
  "revoked": false
}
```

Indexes:
- `{ key_hash: 1 }` — unique, used by classify service on every request
- `{ user_id: 1 }` — list keys for a user
- `{ revoked: 1 }` — filter active keys

---

### `user_configs`

One document per user. Fetched by the classify service to select NER/NEL model
and taxonomy for a request.

```json
{
  "_id": ObjectId,
  "user_id": "usr_abc123",           // → users.user_id
  "ner_type": "SELF_HOSTED_LLM",     // see NER type options below
  "nel_type": "generic",             // see NEL type options below
  "ner_model_name": "tabiya/roberta-base-job-ner",
  "nel_model_name": "all-MiniLM-L6-v2",
  "taxonomy_model_id": "ui1u2jn",    // ESCO taxonomy version ID
  "updated_at": ISODate
}
```

Indexes:
- `{ user_id: 1 }` — unique

---

## API Key Flow

```
Client request
  ↓ x-api-key: sk_live_...
GCP API Gateway
  ↓ validates key exists (GCP-managed keys)
  ↓ forwards request + x-api-key header
Classify Service
  1. hash the key: SHA-256(x-api-key) → key_hash
  2. look up api_keys by key_hash (TTL cache: 5 min)
  3. fetch user_configs by user_id (TTL cache: 5 min)
  4. select NER/NEL model and taxonomy from config
  5. call NER → NEL → return result
```

If no MongoDB is configured (local dev), the classify service uses a default config:
```json
{
  "ner_model_name": "tabiya/roberta-base-job-ner",
  "nel_model_name": "all-MiniLM-L6-v2"
}
```

---

## NER Type Options

| Value | Description |
|-------|-------------|
| `SELF_HOSTED_LLM` | Tabiya-hosted transformer (default) |
| `PARTNER_FINE_TUNED` | Partner-provided fine-tuned model |

## NEL Type Options

| Value | Description |
|-------|-------------|
| `generic` | General ESCO matching (default) |
| `partner_specific` | Custom taxonomy for partner |

---

## Indexes to Create (Atlas UI or script)

```js
// users
db.users.createIndex({ firebase_uid: 1 }, { unique: true });
db.users.createIndex({ user_id: 1 }, { unique: true });

// api_keys
db.api_keys.createIndex({ key_hash: 1 }, { unique: true });
db.api_keys.createIndex({ user_id: 1 });
db.api_keys.createIndex({ revoked: 1 });

// user_configs
db.user_configs.createIndex({ user_id: 1 }, { unique: true });
```
