import sys
import os
import time
import uuid
import logging
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from flask import request, jsonify
from dotenv import load_dotenv
import requests as http_requests

from app.server.common import setup_logging, create_app
from util.job_text import build_input_text, compute_hash

load_dotenv()
setup_logging()
log = logging.getLogger("classify-api")

NER_API_URL = os.getenv("NER_API_URL")
NEL_API_URL = os.getenv("NEL_API_URL")
CLASSIFIER_VERSION = os.getenv("CLASSIFIER_VERSION", "1.0.0")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "50000"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "500"))

app = create_app("classify-api", __name__)

_batches = {}



# Core classification logic

def _classify_text(input_text, options=None):
    """Call NER then NEL and merge results (synchronous, uses requests)."""
    options = options or {}
    entity_types = options.get("extract_entities")
    top_k = options.get("top_k", 5)
    min_similarity = options.get("min_similarity", 0.0)

    start = time.time()

    ner_payload = {"text": input_text}
    if entity_types:
        ner_payload["entity_types"] = entity_types

    ner_resp = http_requests.post(f"{NER_API_URL}/v1/ner", json=ner_payload, timeout=60)
    ner_resp.raise_for_status()
    ner_data = ner_resp.json()
    ner_entities = ner_data.get("entities", [])

    linkable_types = {"occupation", "skill", "qualification"}
    nel_input = [
        {"text": e["surface_form"], "entity_type": e["entity_type"]}
        for e in ner_entities
        if e["entity_type"] in linkable_types
    ]

    linked_map = {}
    nel_metadata = {}
    if nel_input:
        nel_resp = http_requests.post(
            f"{NEL_API_URL}/v1/nel",
            json={"entities": nel_input, "options": {"top_k": top_k, "min_similarity": min_similarity}},
            timeout=60,
        )
        nel_resp.raise_for_status()
        nel_data = nel_resp.json()
        nel_metadata = nel_data.get("metadata", {})

        for item in nel_data.get("linked_entities", []):
            key = (item["input_text"], item["entity_type"])
            linked_map[key] = item["matches"]

    merged_entities = []
    entity_counts = {}

    for entity in ner_entities:
        etype = entity["entity_type"]
        entity_counts[etype] = entity_counts.get(etype, 0) + 1

        merged = {
            "entity_type": etype,
            "surface_form": entity["surface_form"],
            "span": entity["span"],
        }

        key = (entity["surface_form"], etype)
        if key in linked_map:
            merged["linked_entities"] = linked_map[key]

        merged_entities.append(merged)

    processing_time = round((time.time() - start) * 1000, 1)
    input_text_hash = compute_hash(input_text)

    return {
        "classification": {
            "entities": merged_entities,
            "entity_counts": entity_counts,
        },
        "metadata": {
            "classifier_version": CLASSIFIER_VERSION,
            "model_name": ner_data.get("metadata", {}).get("model_name", "unknown"),
            "linker_model": nel_metadata.get("linker_model", "unknown"),
            "processing_time_ms": processing_time,
            "input_text_hash": input_text_hash,
        },
    }



# Single classify endpoint


@app.route("/v1/classify", methods=["POST"])
def classify():
    data = request.get_json(silent=True) or {}

    input_text = build_input_text(data, allow_text_field=True)
    if not input_text:
        return jsonify({"error": "Provide 'text' or 'title'+'description'"}), 400

    if len(input_text) > MAX_TEXT_LENGTH:
        return jsonify({
            "error": f"Text exceeds maximum length ({MAX_TEXT_LENGTH} chars)"
        }), 413

    log.info("Classify request: %d chars", len(input_text))

    try:
        result = _classify_text(input_text, data.get("options"))
    except http_requests.HTTPError as e:
        log.error("Downstream error: %s", e)
        return jsonify({"error": f"Downstream API error: {e}"}), 502

    entity_count = sum(result["classification"]["entity_counts"].values())
    log.info("Classify done: %d entities in %.1fms", entity_count, result["metadata"]["processing_time_ms"])
    return jsonify(result)


# Batch endpoints


def _process_batch(batch_id, jobs, options):
    """Background thread: classify each job and update batch state."""
    batch = _batches[batch_id]

    for i, job in enumerate(jobs):
        input_text = build_input_text(job, allow_text_field=True)
        job_id = job.get("job_id", f"job_{i}")

        if not input_text:
            batch["results"].append({
                "job_id": job_id,
                "status": "error",
                "error": "No classifiable text found",
            })
        elif len(input_text) > MAX_TEXT_LENGTH:
            batch["results"].append({
                "job_id": job_id,
                "status": "error",
                "error": f"Text exceeds {MAX_TEXT_LENGTH} char limit",
            })
        else:
            try:
                result = _classify_text(input_text, options)
                batch["results"].append({
                    "job_id": job_id,
                    "status": "completed",
                    **result,
                })
            except Exception as e:
                log.error("[batch-%s] Job %s failed: %s", batch_id, job_id, e)
                batch["results"].append({
                    "job_id": job_id,
                    "status": "error",
                    "error": str(e),
                })

        batch["processed"] = i + 1

    batch["status"] = "completed"
    batch["completed_at"] = time.time()


@app.route("/v1/classify/batch", methods=["POST"])
def submit_batch():
    data = request.get_json(silent=True) or {}
    jobs = data.get("jobs", [])
    options = data.get("options")

    if not jobs:
        return jsonify({"error": "Field 'jobs' is required and must be non-empty"}), 400

    if len(jobs) > MAX_BATCH_SIZE:
        return jsonify({
            "error": f"Batch too large ({len(jobs)} jobs). Maximum is {MAX_BATCH_SIZE}."
        }), 413

    batch_id = str(uuid.uuid4())[:8]
    _batches[batch_id] = {
        "status": "processing",
        "total": len(jobs),
        "processed": 0,
        "results": [],
        "created_at": time.time(),
        "completed_at": None,
    }

    thread = threading.Thread(
        target=_process_batch, args=(batch_id, jobs, options), daemon=True
    )
    thread.start()

    log.info("Batch %s submitted: %d jobs", batch_id, len(jobs))
    return jsonify({"batch_id": batch_id, "total": len(jobs), "status": "processing"}), 202


@app.route("/v1/batch/<batch_id>/status", methods=["GET"])
def batch_status(batch_id):
    batch = _batches.get(batch_id)
    if not batch:
        return jsonify({"error": "Batch not found"}), 404

    return jsonify({
        "batch_id": batch_id,
        "status": batch["status"],
        "total": batch["total"],
        "processed": batch["processed"],
    })


@app.route("/v1/batch/<batch_id>/results", methods=["GET"])
def batch_results(batch_id):
    batch = _batches.get(batch_id)
    if not batch:
        return jsonify({"error": "Batch not found"}), 404

    return jsonify({
        "batch_id": batch_id,
        "status": batch["status"],
        "total": batch["total"],
        "processed": batch["processed"],
        "results": batch["results"],
    })


# Health / version

@app.route("/v1/health", methods=["GET"])
def health():
    ner_ok = False
    nel_ok = False

    try:
        r = http_requests.get(f"{NER_API_URL}/v1/health", timeout=5)
        ner_ok = r.status_code == 200
    except Exception:
        pass

    try:
        r = http_requests.get(f"{NEL_API_URL}/v1/health", timeout=5)
        nel_ok = r.status_code == 200
    except Exception:
        pass

    overall = "healthy" if (ner_ok and nel_ok) else "degraded"
    return jsonify({
        "status": overall,
        "service": "classify-api",
        "dependencies": {
            "ner_api": "healthy" if ner_ok else "unavailable",
            "nel_api": "healthy" if nel_ok else "unavailable",
        },
    })


@app.route("/v1/version", methods=["GET"])
def version():
    return jsonify({
        "service": "classify-api",
        "version": CLASSIFIER_VERSION,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=os.getenv("FLASK_DEBUG", "0") == "1")
