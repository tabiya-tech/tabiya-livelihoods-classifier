# Inference Pipeline

## Prerequisites

- [Python 3.10 or higher](https://www.python.org/downloads/)
- [Git LFS](https://git-lfs.github.com/)

## Installation

### Using Git LFS

This repository uses Git LFS for handling large files. Before you can use this repository, you need to install and set up Git LFS on your local machine.
See https://git-lfs.com/ for installation instructions.

After Git LFS is set up, follow these steps to clone the repository:

```bash
git clone https://github.com/tabiya-tech/tabiya-livelihoods-classifier.git
```

If you already cloned the repository without Git LFS, run:

```bash
git lfs pull
```

### Install the requirements

In the **root directory** of the project (not the same directory as this README file), run the following commands:

```bash
pip install -r requirements.txt
```

Activate Python and download the NLTK punctuation package to use the sentence tokenizer. You only need to download `punkt` once.

```bash
python
import nltk
nltk.download('punkt')
```

## Usage

To use the entity linker, you need access to the HuggingFace 🤗 entity extraction model. Contact the administrators via [tabiya@benisis.de]. From there, you need to create a read access token to use the model. Find or create your read access token [here](https://huggingface.co/settings/tokens).

### Create the pipeline

```python
from inference.linker import EntityLinker

access_token = "hf_..."
pipeline = EntityLinker(hf_token=access_token, k=5)
```

You can now perform inference on any text with the following code:

```python
text = 'We are looking for a Head Chef who can plan menus.'
extracted = pipeline(text)
print(extracted)
```

### Output

```python
[
  {'type': 'Occupation', 'tokens': 'Head Chef', 'retrieved': ['head chef', 'industrial head chef', 'head pastry chef', 'chef', 'kitchen chef']},
  {'type': 'Skill', 'tokens': 'plan menus', 'retrieved': ['plan menus', 'plan patient menus', 'present menus', 'plan schedule', 'plan engineering activities']}
]
```

## Running the evaluation tests

Load the `Evaluator` class and print the results:

```python
from inference.evaluator import Evaluator

access_token = "hf_..."
results = Evaluator(entity_type='Skill', entity_model='tabiya/roberta-base-job-ner', similarity_model='all-MiniLM-L6-v2', crf=False, evaluation_mode=True, hf_token=access_token)
print(results.output)
```

## Minimum Hardware

- 4 GB CPU/GPU RAM

The code runs on GPU if available. Ensure your machine has CUDA installed if running on GPU.

## Details on the Entity Linker

If you need more precision on using the entity linker, here is detailed information on the parameters and the output.

### Initialization Arguments

- **`entity_model`**: `str`, default: `'tabiya/roberta-base-job-ner'`
  - Path to a pre-trained `AutoModelForTokenClassification` model or an `AutoModelCrfForNer` model. This model is used for entity recognition within the input text.
  
- **`similarity_model`**: `str`, default: `'all-MiniLM-L6-v2'`
  - Path or name of a sentence transformer model used for embedding text. This model converts text into high-dimensional vectors (embeddings) to measure the similarity between different pieces of text.
  - The model `'all-mpnet-base-v2'` is available but not in cache, so it should be used with the parameter **`from_cache`**= `False` at least the first time.
  
- **`crf`**: `bool`, default: `False`
  - A flag to indicate whether to use an `AutoModelCrfForNer` model instead of a standard `AutoModelForTokenClassification`. `CRF` (Conditional Random Field) models are used when the task requires sequential predictions with dependencies between the outputs.
  
- **`hf_token`**: `str`, default: `None`
  - HuggingFace token used for accessing private models. This is necessary as the models are not publicly available and require authentication.
  
- **`evaluation_mode`**: `bool`, default: `False`
  - If set to `True`, the linker will return the cosine similarity scores between the embeddings. This mode is useful for evaluating the quality of the linkages.
  
- **`k`**: `int`, default: `32`
  - Specifies the number of items to retrieve from the reference sets. This parameter limits the number of top matches to consider when linking entities.
  
- **`from_cache`**: `bool`, default: `True`
  - If set to `True`, the precomputed embeddings are loaded from cache to save time. If set to `False`, the embeddings are computed on-the-fly, which requires GPU access for efficiency and can be time-consuming.

- **`output_format`**: `str`, default: `name`
  - Specifies the format of the output for occupations, either 'name' for occupation names or 'esco_code' for ESCO codes.

### Calling Arguments

- **`text`**: `str`
  - An arbitrary job vacancy-related string that the model processes to extract and link entities.
  
- **`linking`**: `bool`, default: `True`
  - Specifies whether the model should perform the entity linking to the taxonomy. If `False`, it might only extract entities without linking them to a predefined taxonomy.

### Output

The output of the `EntityLinker` is a list of dictionaries, where each dictionary represents an identified entity within the input text. Each dictionary contains the following keys:

- **`type`**: The category of the identified entity. The categories of interest are 'Occupation', 'Qualifications', and 'Skill'.
- **`tokens`**: The specific part of the input text that was identified as an entity of the right category.
- **`retrieved`**: A list of related names or ESCO codes retrieved from the reference sets. These items represent the most similar entities or concepts based on the embeddings and similarity calculations.

This structured output enables the application to not only identify key entities within a job description but also to link these entities to the ESCO taxonomy, facilitating better understanding and categorization of the information.