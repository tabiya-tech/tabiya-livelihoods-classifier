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
log = logging.getLogger("ner-api")

NER_MODEL = os.getenv("NER_MODEL", "tabiya/roberta-base-job-ner")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "50000"))

app = create_app("ner-api", __name__)

ner_model = None
_model_load_error = None


def _load_model():
    global ner_model, _model_load_error
    if ner_model is not None or _model_load_error is not None:
        return
    try:
        from inference.ner import NERModel
        ner_model = NERModel(model_name=NER_MODEL)
        log.info("NER model loaded: %s", NER_MODEL)
    except Exception as e:
        _model_load_error = str(e)
        log.error("Failed to load NER model: %s", e)


@app.route("/v1/ner", methods=["POST"])
def extract_entities():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "Field 'text' is required"}), 400

    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"Text exceeds maximum length ({MAX_TEXT_LENGTH} chars)"}), 413

    if ner_model is None:
        return jsonify({"error": _model_load_error or "NER model not loaded"}), 503

    start = time.time()
    try:
        entities = ner_model.extract(text)
    except Exception as e:
        log.error("NER inference failed: %s", e)
        return jsonify({"error": f"Model inference failed: {e}"}), 500

    processing_time = round((time.time() - start) * 1000, 1)

    entity_types = data.get("entity_types")
    if entity_types:
        allowed = {t.lower() for t in entity_types}
        entities = [e for e in entities if e["entity_type"] in allowed]

    log.info("NER done: %d entities in %.1fms", len(entities), processing_time)

    return jsonify({
        "entities": entities,
        "metadata": {
            "model_name": ner_model.model_name,
            "entity_count": len(entities),
            "processing_time_ms": processing_time,
        },
    })


@app.route("/v1/health", methods=["GET"])
def health():
    model_ok = ner_model is not None
    resp = {
        "status": "healthy" if model_ok else "unavailable",
        "service": "ner-api",
        "model_loaded": model_ok,
    }
    if model_ok:
        resp["model_name"] = ner_model.model_name
    if _model_load_error:
        resp["error"] = _model_load_error
    return jsonify(resp), 200 if model_ok else 503


if __name__ == "__main__":
    _load_model()
    app.run(host="0.0.0.0", port=5002, debug=os.getenv("FLASK_DEBUG", "0") == "1")
