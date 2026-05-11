import sys
import os
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from flask import request, jsonify
from dotenv import load_dotenv

from app.server.common import setup_logging, create_app

load_dotenv()
setup_logging()
log = logging.getLogger("nel-api")

LINKER_MODEL = os.getenv("LINKER_MODEL", "all-MiniLM-L6-v2")
MAX_ENTITIES_PER_REQUEST = int(os.getenv("MAX_ENTITIES_PER_REQUEST", "200"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "50"))

app = create_app("nel-api", __name__)

nel_linker = None
_linker_load_error = None


def _load_linker():
    global nel_linker, _linker_load_error
    if nel_linker is not None or _linker_load_error is not None:
        return
    try:
        from inference.nel import NELLinker
        nel_linker = NELLinker(similarity_model=LINKER_MODEL)
        log.info("NEL linker loaded: %s", LINKER_MODEL)
    except Exception as e:
        _linker_load_error = str(e)
        log.error("Failed to load NEL linker: %s", e)


@app.route("/v1/nel", methods=["POST"])
def link_entities():
    data = request.get_json(silent=True) or {}
    entities = data.get("entities", [])
    options = data.get("options", {})

    if not entities:
        return jsonify({"error": "Field 'entities' is required and must be non-empty"}), 400

    if len(entities) > MAX_ENTITIES_PER_REQUEST:
        return jsonify({
            "error": f"Too many entities ({len(entities)}). Maximum is {MAX_ENTITIES_PER_REQUEST}."
        }), 413

    top_k = min(options.get("top_k", 5), MAX_TOP_K)
    min_similarity = options.get("min_similarity", 0.0)

    if nel_linker is None:
        return jsonify({"error": _linker_load_error or "NEL linker not loaded"}), 503

    log.info("NEL request: %d entities, top_k=%d", len(entities), top_k)
    start = time.time()

    try:
        results = nel_linker.link(entities, top_k=top_k, min_similarity=min_similarity)
    except Exception as e:
        log.error("NEL linking failed: %s", e)
        return jsonify({"error": f"Entity linking failed: {e}"}), 500

    processing_time = round((time.time() - start) * 1000, 1)
    log.info("NEL done: %d linked in %.1fms", len(results), processing_time)

    return jsonify({
        "linked_entities": results,
        "metadata": {
            "linker_model": nel_linker.similarity_model_name,
            "taxonomy": "esco",
            "processing_time_ms": processing_time,
        },
    })


@app.route("/v1/health", methods=["GET"])
def health():
    linker_ok = nel_linker is not None
    resp = {
        "status": "healthy" if linker_ok else "unavailable",
        "service": "nel-api",
        "model_loaded": linker_ok,
    }
    if linker_ok:
        resp["linker_model"] = nel_linker.similarity_model_name
    if _linker_load_error:
        resp["error"] = _linker_load_error
    return jsonify(resp), 200 if linker_ok else 503


if __name__ == "__main__":
    _load_linker()
    app.run(host="0.0.0.0", port=5003, debug=os.getenv("FLASK_DEBUG", "0") == "1")
