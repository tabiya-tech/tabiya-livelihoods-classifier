"""Sentence-BERT fine-tuning script (migrated from train/sbert_train.py).

Trains a SentenceTransformer model using MegaBatchMarginLoss on a CSV dataset
that maps job titles to ESCO labels.

Dataset CSV format::

    title,esco_label
    "Head Chef","Chef"
    "Head Chef","Restaurant cook"

Usage::

    python -m training.train_sbert \\
        --dataset ./your_dataset.csv \\
        --model all-MiniLM-L6-v2 \\
        --output ./model-output
"""

import argparse

import pandas as pd
from datasets import Dataset, DatasetDict
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader
from tqdm import tqdm


def df_to_dict(df: pd.DataFrame) -> dict:
    """Convert a title→esco_label CSV DataFrame to a HuggingFace-compatible dict."""
    sentence_set = []
    for title, group in tqdm(df.groupby("title")):
        lista = group["esco_label"].tolist()
        title = title.strip()
        lista.append(title)
        if lista not in sentence_set:
            sentence_set.append(lista)
    return {"set": sentence_set}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a SentenceTransformer model")
    parser.add_argument("--dataset", required=True, help="Path to training CSV file")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model ID")
    parser.add_argument("--output", required=True, help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    dictionary = df_to_dict(df)

    train_dataset = Dataset.from_dict(dictionary)
    dataset_dict = DatasetDict({"train": train_dataset})

    train_examples = []
    train_data = dataset_dict["train"]["set"]
    for example in train_data:
        for j in range(len(example) - 1):
            train_examples.append(InputExample(texts=[example[-1], example[j]]))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    model = SentenceTransformer(args.model)
    train_loss = losses.MegaBatchMarginLoss(model=model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        output_path=args.output,
    )
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
