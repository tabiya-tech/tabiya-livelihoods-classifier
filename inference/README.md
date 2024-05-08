
## Inference Pipeline

python>=3.10
# Installation
Clone the tabiya-livelihoods-classifier with git

```
git clone https://github.com/tabiya-tech/tabiya-livelihoods-classifier.git
```

Navigate to folder

```
cd tabiya-livelihoods-classifier
```

Install the requirements

```
pip install -r requirements.txt
```

Acticate python and download nltk punctuation package to use the sentence tokenizer

```
python
```

```
import nltk
nltk.download('punkt')
```

# Usage

In order to use the entity linker you need to have access to the HuggingFace 🤗 entity extraction model. Feel free to contact the administrators via [tabiya@benisis.de].
From there you need to create a read access token to use the model.  
```python
from inference.linker import EntityLinker
access_token = "hf_..."
text = 'We are looking for a Head Chef who can plan menus.'
pipeline = EntityLinker(entity_model = 'tabiya/bert-base-job-extract', similarity_model = 'all-MiniLM-L6-v2', hf_token = access_token)
extracted = pipeline(text)
print(extracted)
```
Output

```
[{'type': 'Occupation', 'tokens': 'Head Chef', 'retrieved': ['3434.1.1', '3434.1.2.1', '3434.1', '5120.1']}, {'type': 'Skill', 'tokens': 'plan menus', 'retrieved': ['plan menus', 'plan patient menus', 'present menus', 'plan schedule', 'plan engineering activities']}]
```

Find or create your read access token [here](https://huggingface.co/settings/tokens)

# Minimum Hardware

8 GB CPU/ GPU RAM 

The code runs in GPU, if available. Be sure your machine has cuda installed, if running on GPU.
