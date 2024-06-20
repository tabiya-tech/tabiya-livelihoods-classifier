# Training

Train your entity extraction model using PyTorch.

In order to use the entity linker, you need access to the HuggingFace 🤗 entity extraction model. Feel free to contact the administrators via [tabiya@benisis.de]. From there, you need to create a read access token to use the training dataset. 

Find or create your read access token [here](https://huggingface.co/settings/tokens).

First, activate the virtual environment as explained [here](../README.md).

## Train an Entity Extraction Model

Configure the necessary hyperparameters in the [config.json file](config.json). The defaults are:

```json
{
    "model_name": "bert-base-cased",
    "crf": false,
    "dataset_path": "tabiya/job_ner_dataset",   
    "label_list": ["O", "B-Skill", "B-Qualification", "I-Domain", "I-Experience", "I-Qualification", "B-Occupation", "B-Domain", "I-Occupation", "I-Skill", "B-Experience"],
    "model_max_length": 128,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 4,
    "weight_decay": 0.01,
    "save": false,
    "output_path": "bert_job_ner",
    "access_token": "yourhftoken"
}
```

To train the model, run the following script:

```sh
python train.py
```

## Train an Entity Similarity Model

Configure the necessary hyperparameters in the `sbert_train` function in the [sbert_train.py file](sbert_train.py):

```python
sbert_train(model_id='all-MiniLM-L6-v2', dataset_path='your/dataset/path', output_path='your/output/path')
```

To train the similarity model, run the following script:

```sh
python sbert_train.py
```