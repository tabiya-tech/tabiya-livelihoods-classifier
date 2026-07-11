# Tabiya Pipeline Plugins

The classifier runs job text through a **pipeline of plugins**. Each plugin is a
small, single-purpose transform (extract entities, link them to a taxonomy, read
input, write output). A pipeline is a linear chain of them:

```
text_input  →  ner  →  nel  →  results
 (Source)     (Core)  (Core)   (Sink)
 None→RawText  RawText  Entities  LinkedEntities
              →Entities →Linked   →None
```

This document explains **how the plugin system works** and **how to build your
own plugin**. If you just want the recipe, jump to
[Build your own plugin](#build-your-own-plugin).

---

## The big picture

Every plugin is an **HTTP microservice endpoint**. It speaks one small contract:

| Endpoint | Purpose |
|---|---|
| `GET /plugin/{id}/manifest` | Static metadata: name, category, input/output types, config schema. |
| `POST /plugin/{id}/invoke` | Do the work: take a typed input + config, return a typed output. |
| `GET /plugin/{id}/health` | Is this plugin able to serve invokes right now? |

You never implement those endpoints by hand. You write two things — a
**manifest** and a **core function** — and a shared helper
(`make_http_adapter`) turns them into that HTTP contract for you.

Plugins don't run as separate processes each. They're grouped into **bundles**
(one Cloud Run service per bundle), and the orchestrator (`classify_v2`)
discovers and drives them.

```
┌─────────────────────────────────────────────────────────────┐
│ classify_v2  (orchestrator)                                   │
│   • registry:  reads catalog.json, fetches each /manifest     │
│   • executor:  walks a pipeline, POSTs /invoke stage by stage │
└───────────────┬───────────────────────────┬──────────────────┘
                │ HTTP /plugin/*/invoke      │
        ┌───────▼────────┐          ┌────────▼───────┐
        │ tabiya_core    │          │ tabiya_io      │   ← bundles
        │  ner, nel      │          │  text_input,   │     (Cloud Run
        │                │          │  json*, results│      services)
        └────────────────┘          └────────────────┘
                        each mounts N plugins at /plugin/{id}
```

---

## The four layers

### 1. The contract — `backend/tabiya_plugin_contracts/`

The shared vocabulary every plugin and the orchestrator agree on. Key pieces:

- **`SlotType`** — the five typed payloads that flow between stages:
  `None`, `RawText`, `RawTextStream`, `Entities`, `LinkedEntities`. Each has a
  Pydantic model (`RawText{text}`, `Entities{entities[], source_text}`, …).
- **`slot_accepts(producer, consumer)`** — slot compatibility. Exact match, plus
  one subtype rule: `Entities` may feed a `LinkedEntities` input (a
  `LinkedEntity` is an `Entity` with optional `matches`), which lets a pipeline
  end on NER output without an NEL stage.
- **`Manifest`** — the static description of a plugin (see below).
- **`Context` / `InvokeRequest` / `InvokeResponse` / `ErrorEnvelope`** — the
  runtime wire types.
- **`ErrorCode`** — the six error codes a plugin may raise:
  `BAD_INPUT` (400), `CONFIG_INVALID` (400), `TIMEOUT` (504),
  `UPSTREAM_UNAVAILABLE` (503), `UNAVAILABLE` (503), `PLUGIN_INTERNAL` (500).
- **`CONTRACT_VERSION`** — semver. The orchestrator compares its **major** against
  each plugin's declared version at registration; a mismatch marks the plugin
  unavailable rather than risking a shape mismatch.

### 2. The adapter — `make_http_adapter(manifest, invoke_fn, health_fn)`

This is the machine that turns your manifest + core into the HTTP contract. For
every `/invoke` it:

1. Parses the request envelope, then validates `input` against the model for
   `manifest.input_slot.type`. Bad shape → `BAD_INPUT`.
2. Calls your `invoke_fn(typed_input, config, context)` under a hard
   `asyncio.wait_for(timeout_ms)`. Overrun → `TIMEOUT` (504).
3. Maps any `PluginError` you raise to its `ErrorCode` + HTTP status; anything
   unexpected becomes `PLUGIN_INTERNAL` (500).
4. Validates your **output** against `manifest.output_slot.type` before
   returning it — a plugin that returns the wrong shape fails loudly.

You never edit this file. You just hand it your two pieces.

### 3. The plugin — a manifest + a core function

A plugin is a small package with three files:

```
plugins/<your_plugin>/
  manifest.py    # MANIFEST = Manifest(...)   — static metadata
  core.py        # async def invoke(input, config, context) -> OutputSlot
  __init__.py    # re-export MANIFEST + invoke
```

**`invoke` signature** the adapter expects:

```python
async def invoke(input: InputSlotModel, config: dict, context: Context) -> OutputSlotModel
```

- `input` — already parsed & validated into the input slot model (e.g. `RawText`).
- `config` — the stage's config dict (validated against your `config_schema`
  only at invoke time, by you if you choose — see [Config](#config--the-config_schema)).
- `context` — `request_id`, `user_id`, `pipeline_id`, `stage_index`, `deadline_ms`.
- return the output slot model (e.g. `Entities`). You may also return
  `(output, metadata_dict)` to attach free-form per-stage metadata.

### 4. The bundle — a Cloud Run service that hosts plugins

A bundle is a thin FastAPI app that mounts every installed plugin. See
`tabiya_io/tabiya_io/main.py`: it loops `INSTALLED_PLUGINS` and, for each,
`app.include_router(make_http_adapter(manifest, invoke, health), prefix=f"/plugin/{id}")`
behind a shared `require_identity_token` auth dependency. There are two bundles
today — `tabiya_core` (CPU-heavy: NER, NEL) and `tabiya_io` (cheap I/O: sources,
sinks) — so they scale independently.

---

## How the orchestrator finds and runs plugins

### Discovery — the registry

`classify_v2` ships a **`catalog.json`**
(`classify_v2/app/pipelines/registry/catalog.json`) listing every plugin:

```json
{ "plugin_id": "tabiya.source.json.v1",
  "url_env": "TABIYA_IO_BUNDLE_URL",
  "path": "/plugin/tabiya.source.json.v1" }
```

At startup (and every 5 min) the registry resolves each entry's bundle URL from
its `url_env` env var, fetches `{url}/manifest`, validates the contract version,
and caches the `Manifest`. A plugin whose manifest declares
`x-tabiya-coming-soon: true` is cached but marked **coming-soon** (shown in the
palette, greyed, rejected by the validator).

### Execution — the executor

`POST /v2/classify` resolves the user's pipeline, then the **executor**
(`classify_v2/app/pipelines/executor/executor.py`) walks the stages in order:

1. Start with a `None` slot as the first input.
2. For each stage: build `{context, config, input}`, POST it to
   `{bundle_url}/plugin/{id}/invoke` (with a GCP identity token in prod), enforce
   the manifest's `timeout_ms`, and feed the returned `output` in as the next
   stage's `input`.
3. Snapshot the `LinkedEntities` (or `Entities`, if NER-only) payload for the
   response, and emit one structured log line per stage.

The executor never knows what any slot *contains* — it just passes `dict`s
through. Type safety is enforced at each plugin's adapter boundary.

---

## Build your own plugin

Worked example: a `tabiya.transform.uppercase.v1` transform that upper-cases the
text (RawText → RawText). Adjust category/slots for your real plugin.

### 1. Create the package

`backend/plugin_bundles/tabiya_io/tabiya_io/plugins/uppercase/`

**`manifest.py`**
```python
from tabiya_plugin_contracts import Manifest, PluginCategory, Slot, SlotType

MANIFEST = Manifest(
    plugin_id="tabiya.transform.uppercase.v1",   # dotted, version-suffixed, unique
    name="Uppercase",
    version="0.1.0",
    category=PluginCategory.TRANSFORM,            # source | core | transform | sink
    summary="Upper-cases the incoming text.",
    detail="demo transform",
    icon="filter",                                # a frontend icon key
    input_slot=Slot(type=SlotType.RAW_TEXT),
    output_slot=Slot(type=SlotType.RAW_TEXT),
    config_schema={                               # JSON-Schema subset for the editor form
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    timeout_ms=5_000,
)
```

**`core.py`**
```python
from tabiya_plugin_contracts import Context, RawText

async def invoke(input: RawText, config: dict, context: Context) -> RawText:
    return RawText(text=input.text.upper())
```

Raise a `PluginError` subclass (from `tabiya_plugin_contracts.adapters.http`) for
expected failures — `ConfigInvalidError`, `BadInputError`,
`UpstreamUnavailableError`, `UnavailableError`. Anything else becomes a 500.

**`__init__.py`**
```python
from .core import invoke
from .manifest import MANIFEST
__all__ = ["MANIFEST", "invoke"]
```

### 2. Register it in the bundle

`tabiya_io/tabiya_io/plugins/__init__.py` — add it to `INSTALLED_PLUGINS`:
```python
from .uppercase import MANIFEST as UPPERCASE_MANIFEST, invoke as uppercase_invoke
INSTALLED_PLUGINS = [
    ...,
    (UPPERCASE_MANIFEST, uppercase_invoke, None),   # (manifest, invoke, optional health_fn)
]
```

### 3. Add it to the orchestrator catalog

`classify_v2/app/pipelines/registry/catalog.json`:
```json
{ "plugin_id": "tabiya.transform.uppercase.v1",
  "url_env": "TABIYA_IO_BUNDLE_URL",
  "path": "/plugin/tabiya.transform.uppercase.v1" }
```

### 4. Test it

Mount just your plugin with the adapter and drive it over HTTP — this is exactly
what the bundle does, so it's a faithful test. See any
`tabiya_io/tests/test_*_plugin.py`:
```python
from tabiya_plugin_contracts.adapters.http import make_http_adapter
app = FastAPI(); app.include_router(make_http_adapter(MANIFEST, invoke), prefix=f"/plugin/{MANIFEST.plugin_id}")
# POST {context, config, input:{text:"hi"}} to /plugin/.../invoke → expect output {text:"HI"}
```

That's it. Rebuild/redeploy the bundle, and the plugin appears in the palette on
the next registry refresh.

---

## Reference

### Categories & slots

- **Source** — `input_slot = None`, produces the first real payload
  (`text_input`: None→RawText, `json_entities`: None→Entities).
- **Core** — the heavy transforms (`ner`: RawText→Entities, `nel`:
  Entities→LinkedEntities).
- **Transform** — same-shape or reshaping middle steps.
- **Sink** — `output_slot = None`, consumes the final payload (`results`:
  LinkedEntities→None).

A pipeline must start with a Source and end with a Sink; adjacent stages must be
slot-compatible (`slot_accepts`).

### Config — the `config_schema`

`config_schema` is a small JSON-Schema object that drives the editor's config
form. Field extensions:
- `"x-source": "/v2/nel/models"` — render a dropdown whose options come from
  that endpoint (the orchestrator's options proxy resolves `/v2/nel/*` against
  the NEL service).
- `"default": ...` — pre-filled when a stage is dropped.

Note: per-stage config is **not** validated at pipeline-save time (you can build
a chain and choose models later). It's validated at invoke time by your plugin.

### Coming-soon plugins

Ship a real manifest with `**{"x-tabiya-coming-soon": True}` and a stub `invoke`
that raises `UnavailableError` + a `health` returning `DOWN`. It shows in the
palette (greyed, undroppable) and the validator rejects pipelines that use it.
See `tabiya_io/tabiya_io/plugins/scraper/` for the pattern.

### Auth

In production every bundle mounts plugins behind `require_identity_token`; the
orchestrator attaches a GCP identity token per invoke. Locally
(`TARGET_ENVIRONMENT_TYPE=local`) auth is bypassed.

### Contract versioning

Bump `CONTRACT_VERSION` (in `tabiya_plugin_contracts/version.py`) only on a
breaking change to the manifest / invoke / slot models. The adapter auto-stamps
each manifest with it; the registry refuses plugins whose **major** differs.
```
