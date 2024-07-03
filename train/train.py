from dotenv import load_dotenv
import os, sys

# Load environment variables from the .env file
load_dotenv(verbose=True)

# Add the parent directory to the system path
self_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(self_path, '../'))

from util.transformersCRF import BertCrfForNer
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer, AutoModelForTokenClassification
import evaluate
import numpy as np
from seqeval.metrics import classification_report
from nervaluate import Evaluator
from seqeval.scheme import IOB2
from util.utilfunctions import Config


# Hyperparameters
config = Config(os.path.join(self_path, 'config.json')) # reads all hyperparameters and sets them as attributes on the config object
MODEL_NAME = config.model_name #tested on roberta-base, jjzha/esco-xlm-roberta-large and jjzha/jobbert-base-cased
USE_CRF = config.crf
dataset_path = config.dataset_path
access_token = os.getenv('HF_TOKEN')
custom_dataset = load_dataset(dataset_path, token = access_token)
label_list = config.label_list
if USE_CRF:
    label_list.append('X')
label2id = {tag:idx for idx, tag in enumerate(label_list)}
id2label = {idx:tag for idx, tag in enumerate(label_list)}
MODEL_MAX_LENGTH = config.model_max_length
BATCH_SIZE = config.batch_size
LEARNING_RATE = config.learning_rate
EPOCHS = config.epochs
WEIGHT_DACAY = config.weight_decay
NUMBER_OF_TAGS = len(label_list)

#Preprocess

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=MODEL_MAX_LENGTH, add_prefix_space=True)
#tokenize method for fast tokenizers. If CRF decoder is used, we levarage the special mask argument of huggingface to pass it as the mask in the CRF module.
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    special_masks = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        special_mask = []
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
              if USE_CRF: #Set the [CLS] and [SEP] tokens to the 11th label
                special_mask.append(0)
                label_ids.append(11)
              else:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
              label_ids.append(label[word_idx])  # Only label the first token of a given word.
              if USE_CRF:
                special_mask.append(0)
            else:
              if USE_CRF:
                special_mask.append(1)
                label_ids.append(11)
              else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        special_masks.append(special_mask)
    if USE_CRF:
      tokenized_inputs["special_tokens_mask"] =  special_masks

    tokenized_inputs["labels"] = labels

    return tokenized_inputs

tokenized_custom_dataset = custom_dataset.map(tokenize_and_align_labels, batched=True)


#Create the datacollator. If using the CRF we set the padding mode to 'max_length' and the padding token to the 11th index.
if USE_CRF:
  data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer,padding='max_length', max_length=MODEL_MAX_LENGTH, label_pad_token_id = 11)
else:
  data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


#Evaluate
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    if USE_CRF:
      number = NUMBER_OF_TAGS
    else:
      number = -100
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != number] #
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != number] #
        for prediction, label in zip(predictions, labels)
    ]
    #Print the loose evaluation of NERevaluate
    evaluator = Evaluator(true_labels, true_predictions, tags=['Skill', 'Qualification', 'Domain', 'Experience', 'Occupation'], loader="list")
    results2, results_by_tag = evaluator.evaluate()
    print(classification_report(true_labels, true_predictions, scheme=IOB2))
    print('--------------------------------------')
    print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))
    results = seqeval.compute(predictions=true_predictions, references=true_labels, mode='strict', scheme='IOB2')
    print('--------------------------------------')
    print(results2, results_by_tag)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

if USE_CRF:
  #bert: BertCrfForNer,  roberta: RoBertaCrfForNer
  model = BertCrfForNer.from_pretrained(MODEL_NAME, num_labels=NUMBER_OF_TAGS)
else:
  model = AutoModelForTokenClassification.from_pretrained(
      MODEL_NAME, num_labels=NUMBER_OF_TAGS, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
  )

from transformers import TrainingArguments, Trainer, AutoModelForTokenClassification
training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=WEIGHT_DACAY,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    optim="adamw_torch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_custom_dataset["train"],
    eval_dataset=tokenized_custom_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


if __name__=='__main__':
    trainer.train()

    trainer.evaluate(tokenized_custom_dataset['test'])
    if config.save:
        trainer.save_model(os.path.join(self_path, config.output_path))