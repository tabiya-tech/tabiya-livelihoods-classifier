# Tabiya Livelihoods Classifier

Extracts **occupations**, **skills**, **qualifications**, **experience**, and **domain** entities from job text and links them to the [ESCO taxonomy](https://esco.ec.europa.eu/).

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Git](https://git-scm.com/) (v2.37+) with [Git LFS](https://git-lfs.com/)
- **HuggingFace Token**

---

## Setup

```bash
git lfs install
git clone https://github.com/tabiya-tech/tabiya-livelihoods-classifier.git
cd tabiya-livelihoods-classifier

cp .env.example .env
# Edit .env and set: HF_TOKEN=hf_your_token_here

docker compose up -d
```

First startup takes several minutes (image build + ~500 MB model download). Monitor with `docker compose logs -f`.

Once ready, verify:

```bash
curl http://localhost:5001/v1/health
```

```json
{
  "status": "healthy",
  "service": "classify-api",
  "dependencies": {
    "ner-api": "healthy",
    "nel-api": "healthy"
  }
}
```

---

## Usage

### Classify a job

```bash
curl -X POST http://localhost:5001/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Head Chef",
    "description": "We are looking for an experienced Head Chef to plan menus, manage kitchen staff, and ensure food quality."
  }'
```

You can also pass a single `"text"` field instead of `title` + `description`.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `top_k` | 5 | ESCO matches per entity |
| `min_similarity` | 0.0 | Minimum cosine similarity (0.0–1.0) |
| `extract_entities` | all | Filter entity types, e.g. `["occupation", "skill"]` |

### Batch

```bash
curl -X POST http://localhost:5001/v1/classify/batch \
  -H "Content-Type: application/json" \
  -d '{
    "jobs": [
      {"job_id": "001", "title": "Nurse", "description": "Provide patient care."},
      {"job_id": "002", "text": "Electrician. Install and maintain electrical systems."}
    ]
  }'
```

Returns a `batch_id` — poll `/v1/batch/{id}/status`, fetch results at `/v1/batch/{id}/results`.

### Response format

```json
{
  "classification": {
    "entities": [
      {
        "entity_type": "occupation",
        "surface_form": "Head Chef",
        "span": {"start": 0, "end": 9},
        "linked_entities": [
          {
            "label": "head chef",
            "code": "3434.2",
            "uri": "http://data.europa.eu/esco/occupation/...",
            "similarity_score": 0.91,
            "taxonomy": "esco"
          }
        ]
      },
      {
        "entity_type": "skill",
        "surface_form": "plan menus",
        "span": {"start": 52, "end": 62},
        "linked_entities": [
          {
            "label": "plan menus",
            "similarity_score": 0.88,
            "taxonomy": "esco"
          }
        ]
      }
    ],
    "entity_counts": {"occupation": 1, "skill": 3}
  },
  "metadata": {
    "classifier_version": "1.0.0",
    "model_name": "tabiya/roberta-base-job-ner",
    "linker_model": "all-MiniLM-L6-v2",
    "processing_time_ms": 245.3,
    "input_text_hash": "abc123..."
  }
}
```

Entity types: `occupation`, `skill`, `qualification`, `experience`, `domain`. Only occupation, skill, and qualification get `linked_entities`.

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/classify` | Classify a single job |
| `POST` | `/v1/classify/batch` | Submit a batch (async) |
| `GET` | `/v1/batch/{id}/status` | Poll batch progress |
| `GET` | `/v1/batch/{id}/results` | Fetch batch results |
| `GET` | `/v1/health` | Health check |
| `GET` | `/v1/version` | Version info |

Interactive docs at `http://localhost:5001/docs`.

---

## Manual setup (without Docker)

Requires **Python 3.10+** and [Poetry](https://python-poetry.org/).

```bash
python3 -m venv venv && source venv/bin/activate
poetry install --sync
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
export HF_TOKEN=hf_your_token_here
```

Start each in a separate terminal:

```bash
uvicorn app.server.ner_server:app --host 0.0.0.0 --port 5002
uvicorn app.server.nel_server:app --host 0.0.0.0 --port 5003
uvicorn app.server.classify_server:app --host 0.0.0.0 --port 5001
```

---

## Stopping

```bash
docker compose down
```
