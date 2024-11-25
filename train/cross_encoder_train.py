# -*- coding: utf-8 -*-
"""
This examples trains a CrossEncoder for the Quora Duplicate Questions Detection task. A CrossEncoder takes a sentence pair
as input and outputs a label. Here, it output a continuous labels 0...1 to indicate the similarity between the input pair.

It does NOT produce a sentence embedding and does NOT work for individual sentences.

Usage:
python cross_encoder_training.py

Original repository: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/cross-encoder/training_quora_duplicate_questions.py
"""

from torch.utils.data import DataLoader
import math
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample

from datetime import datetime

import pandas as pd
from huggingface_hub import hf_hub_download
import pandas as pd
HF_TOKEN = "hf_..."
REPO_ID = "tabiya/occupation_titles_esco"
FILENAME = "cross_encoder_training.csv"


# Read the cross encoder dataset. This dataset was created matching the appropriate ESCO occupation, both preferred and alternative labels, with the corresponding Hahu annotation.

df = pd.read_csv(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset", token=HF_TOKEN)
)

df = df.sample(frac = 1)

#Use 90% of entries as training data, use the last 10000 entries as test set and the rest as validation set.
df_train = df.iloc[:int(len(df)*0.9),:]
df_dev = df.iloc[int(len(df)*0.9):-10000,:]
df_test = df.iloc[-10000:,:]

#Configure training samples
train_samples = []
for idx in df_train.index:
    train_samples.append(InputExample(texts=[df_train["title_hahu"][idx], df_train["title_esco"][idx]], label=int(df_train["label"][idx])))
    train_samples.append(InputExample(texts=[df_train["title_esco"][idx], df_train["title_hahu"][idx]], label=int(df_train["label"][idx])))

#Configure validations samples
dev_samples = []
for idx in df_dev.index:
    dev_samples.append(InputExample(texts=[df_dev["title_hahu"][idx], df_dev["title_esco"][idx]], label=int(df_dev["label"][idx])))

#Configuration
train_batch_size = 32
num_epochs = 1
model_save_path = "output/training_hahu-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


#We use bert-base-cased with a single label, i.e., it will output a value between 0 and 1 indicating the similarity of the two questions
model = CrossEncoder("bert-base-cased", num_labels=1)

#We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

#We add an evaluator, which evaluates the performance during training
evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name="HAHU-dev")

#Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up

if __name__=='__main__':
#Τrain the model
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=5000,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
    )