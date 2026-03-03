from tqdm import tqdm
import pandas as pd
from datasets import Dataset, DatasetDict
from sentence_transformers import InputExample, losses, SentenceTransformer
from torch.utils.data import DataLoader


def df_to_dict(df):
  """Convert a pandas DataFrame of title→esco_label mappings into a HuggingFace Dataset-compatible dict."""
  sentence_set = []
  for title, group in tqdm(df.groupby('title')):
    lista = group['esco_label'].tolist()
    if title[0] == ' ':
      title = title[1:]
    if title[-1] == ' ':
      title = title[:-1]
    lista.append(title)
    if lista not in sentence_set:
      sentence_set.append(lista)

  dataset_dict = {
      "set": sentence_set
  }
  return dataset_dict


def sbert_train(model_id: str, dataset_path : str,  output_path: str) -> None:
    """Fine-tune a sentence transformer on title↔ESCO label pairs."""
    df = pd.read_csv(dataset_path)
    dictionary = df_to_dict(df)

    train_dataset = Dataset.from_dict(dictionary)
    dataset_dict = DatasetDict({"train": train_dataset})

    train_examples = []
    train_data = dataset_dict['train']['set']

    for i in range(len(train_data)):
      example = train_data[i]
      for j in range(len(example)):
        if j != len(example)-1:
          train_examples.append(InputExample(texts=[example[-1], example[j]]))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

    model = SentenceTransformer(model_id)

    train_loss = losses.MegaBatchMarginLoss(model=model)

    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, output_path=output_path)

if __name__=='__main__':
  #TODO: Implement configuration file to train with adjustable hyperparameters.
  sbert_train(model_id = 'all-MiniLM-L6-v2', dataset_path='your/dataset/path', output_path='your/output/path')
