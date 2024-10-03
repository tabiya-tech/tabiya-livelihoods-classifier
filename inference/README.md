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

## Minimum Hardware

- 4 GB CPU/GPU RAM

The code runs on GPU if available. Ensure your machine has CUDA installed if running on GPU.