from tqdm import tqdm
import pandas as pd
from datasets import Dataset, DatasetDict
from sentence_transformers import InputExample, losses, SentenceTransformer
from torch.utils.data import DataLoader


def df_to_dict(df):
  """
  Function that converts pandas dataframe to dictionary for HuggingFace Dataset. It also removes blank spaces from the start and end of titles.
  """
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

  # Create a dictionary with the required format for Hugging Face dataset
  dataset_dict = {
      "set": sentence_set
  }
  return dataset_dict


def sbert_train(model_id: str, dataset_path : str,  output_path: str) -> None:
    """
    Perform Sentence transfromer training.
    Args: 
    model_id: sentence transformer id from HuggingFace
    dataset_path: the local path that the dataset is stored
    output_path: the path to store the model
    """
    df = pd.read_csv(dataset_path)
    dictionary = df_to_dict(df)

    train_dataset = Dataset.from_dict(dictionary)
    dataset_dict = DatasetDict({"train": train_dataset})

    train_examples = []
    train_data = dataset_dict['train']['set']
    #Create tuples from the hahu title and esco preferred and alternative labels
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
