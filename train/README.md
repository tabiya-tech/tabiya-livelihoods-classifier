# Training

Train your entity extraction model using PyTorch.
## Installation
Clone the tabiya-livelihoods-classifier with git

```
git clone https://github.com/tabiya-tech/tabiya-livelihoods-classifier.git
```

Navigate to folder

```
cd tabiya-livelihoods-classifier
```

Basic modules

```
pip install torch transformers datasets evaluate seqeval nervaluate
```

Access the latest accelerate version

```
pip install transformers[torch]
```
## Train an Entity Extraction Model
Configure the necessary hyperparameters in the config.json file. Defaults are:

```javascript 
{
    "model_name":"bert-base-cased",
    "crf": false,
    "dataset_path":"tabiya/job_ner_dataset",   
    "label_list" : [ "O","B-Skill","B-Qualification","I-Domain","I-Experience","I-Qualification","B-Occupation","B-Domain","I-Occupation","I-Skill","B-Experience"],
    "model_max_length" : 128,
    "batch_size" : 32,
    "learning_rate" : 1e-4,
    "epochs" : 4,
    "weight_decay": 0.01,
    "save":false,
    "output_path":"bert_job_ner",
    "access_token": "yourhftoken"
}
```

In order to use the entity linker you need to have access to the HuggingFace 🤗 entity extraction model. Feel free to contact the administrators via [tabiya@benisis.de].
From there you need to create a read access token to use the training dataset.  

Run the script

```
python train.py
```
