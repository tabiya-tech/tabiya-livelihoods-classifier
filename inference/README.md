# Inference Pipeline
The inference pipeline extracts occupations and skills from a job description and matches them to the most similar entities in the ESCO taxonomy.

## Usage

First, activate the virtual environment as explained [here](../README.md#set-up-virtualenv). 

Then, `start python interpreter in the root directory` and run the following commands:

Load the `EntityLinker` class and create an instance of the class,
then perform inference on any text with the following code:

```python
from inference.linker import EntityLinker
pipeline = EntityLinker(k=5)
text = 'We are looking for a Head Chef who can plan menus.'
extracted = pipeline(text)
print(extracted)
```

After running the commands above, you should see the following output:

```js
[
  {'type': 'Occupation', 'tokens': 'Head Chef', 'retrieved': ['head chef', 'industrial head chef', 'head pastry chef', 'chef', 'kitchen chef']},
  {'type': 'Skill', 'tokens': 'plan menus', 'retrieved': ['plan menus', 'plan patient menus', 'present menus', 'plan schedule', 'plan engineering activities']}
]
```

## Running the evaluation tests

Load the `Evaluator` class and print the results:

```python
from inference.evaluator import Evaluator

results = Evaluator(entity_type='Skill', entity_model='tabiya/roberta-base-job-ner', similarity_model='all-MiniLM-L6-v2', crf=False, evaluation_mode=True)
print(results.output)
```

## NER: RoBERTa vs LLM

Entity extraction can use either the default transformer (RoBERTa) or a Vertex AI LLM.

- **Config:** `inference/config.json` supports `ner_type` (`"roberta"` or `"llm"`) and `llm_model` (e.g. `gemini-1.5-pro`). Env vars override: `NER_TYPE`, `LLM_MODEL`, `VERTEX_PROJECT`, `VERTEX_API_REGION`, `GOOGLE_APPLICATION_CREDENTIALS`.
- **RoBERTa (default):** Set `ner_type` to `roberta` or leave unset. Uses `entity_model` (e.g. `tabiya/roberta-base-job-ner`).
- **LLM:** Set `ner_type` to `llm`. Requires `VERTEX_PROJECT` and, for local auth, `GOOGLE_APPLICATION_CREDENTIALS`. Optional: `VERTEX_API_REGION` (default `us-west1`), `LLM_MODEL` (default `gemini-1.5-pro`).

Example with LLM NER:

```python
from inference.linker import EntityLinker
# Via config: set inference/config.json "ner_type" to "llm", or:
pipeline = EntityLinker(ner_type="llm", llm_model="gemini-1.5-flash")
```

## Minimum Hardware

- 4 GB CPU/GPU RAM

The code runs on GPU if available. Ensure your machine has CUDA installed if running on GPU.