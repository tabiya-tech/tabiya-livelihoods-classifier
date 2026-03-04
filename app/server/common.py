import logging

from flask import Flask
from flask_cors import CORS

LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def create_app(name: str, import_name: str) -> Flask:
    """Create a Flask app with CORS enabled (same pattern as matching.py)."""
    app = Flask(import_name)
    CORS(app, resources={r"/*": {"origins": "*"}})
    return app
