# Web Application for Job Description Analysis

This is a Flask-based API for analyzing job descriptions and predicting relevant occupations, skills, and qualifications using an entity linking model.

## Usage

First, activate the virtual environment as explained [here](../README.md#install-the-dependencies). Then, run the following command in python in the `root` directory:

### Running the API

**Run the Flask application**:

```bash
python app/server/matching.py
```

Or set the Flask application environment variable and use the Flask command:

```bash
export FLASK_APP=app/server/matching.py
flask run --host=0.0.0.0 --port=5001
```

## Example Usage

1. **Open the browser** and navigate to `http://127.0.0.1:5001/`.

2. **Paste a job description** into the provided text area.

3. **Click the "Analyze Job" button** to send the job description to the `/match` endpoint.

4. **View the results** under "Predicted Occupations," "Predicted Skills," and "Predicted Qualifications."

---

## Classification API (NER + NEL + Classify)

A microservice-style API that splits the pipeline into three Flask servers:

| Service | Port | Description |
|---------|------|-------------|
| NER API | 5002 | Extracts entity spans from job text |
| NEL API | 5003 | Links entities to ESCO taxonomy via embedding similarity |
| Classify API | 5001 | Orchestrator — calls NER then NEL and merges results |

### Running the services

```bash
# Terminal 1
python app/server/ner_server.py

# Terminal 2
python app/server/nel_server.py

# Terminal 3
python app/server/classify_server.py
```

Or with Docker:

```bash
docker compose up --build
```

### Quick test

```bash
curl -X POST http://localhost:5001/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Head Chef with experience in menu planning."}'
```

### Change Stream Worker

Watches MongoDB `raw-jobs` for new documents and classifies them automatically:

```bash
python app/worker/change_stream_worker.py
python app/worker/change_stream_worker.py --backfill   # process existing unclassified jobs
python app/worker/change_stream_worker.py --dry-run    # log only, no writes
```

Requires `APPLICATION_MONGODB_URI` in `.env` (see `.env.example`).

