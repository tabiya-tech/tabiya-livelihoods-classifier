"""Pipelines feature module.

Composition of a linear pipeline of plugin nodes. See tmp/step-11-design.md
for the full architecture. In v1:

  * `registry/` — resolves plugin URLs, fetches manifests, caches them.
  * `routes/` — read-only plugin catalog endpoints (11.3), CRUD (11.5).
  * `repository/` — pipelines Mongo collection (11.4).
  * `service/` — validator + service (11.5).
  * `executor/` — linear stage walker used by /v2/classify (11.6).
"""
