"""Read-only /v2/plugins routes.

The catalog + registry live upstream; these routes just expose them over
HTTP for the frontend palette + editor to consume. CRUD for pipelines
lives in a separate module (11.5).
"""
