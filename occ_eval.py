import numpy as np
from entity_linker import customPipeline
import pandas as pd

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
    if not actual:
        return 0.0
    return score / min(len(actual), k)

def run_eval(model : str, crf : bool)  -> None:
    """
    Evaluation code for the occupations dataset that computes the MAP@1 and MAP@5 metrics for the entity linker.
    Args: 
        model: Path to an AutoModelForTokenClassification HuggingFace or a BertForCrf custom model 
        crf: Boolean variable used to define whether the entity model uses a CRF decoder or not
    """
    #Read the test set as dataframe
    df = pd.read_csv('hahu_test.csv')
    general_classification_list = ['Community, Social Services and Public Administration',
            'Education',
            'Financial, Insurance and Business Services',
            'Health',
            'Manufacturing and Construction',
            'Manufacturing, Construction and Engineering',
            'Professional, Scientific and Technical',
            'Restaurant and Hospitality Services',
            'Sales and Services',
            'Transport, Storage and ICT',
            'Wholesale and Retail Trade']
    print(f'Running evaluation on {model} model') 
    sum_1 = 0
    sum_2 = 0
    overall_den = 0
    #Define the entity linking pipeline
    custom_pipeline = customPipeline(entity_model=model, similarity_model='all-MiniLM-L6-v2', crf=crf)
    #Iterate through the general classification list to report the MAP on each class. 
    for item in general_classification_list:
      sum_of_ap1 = 0
      sum_of_ap5 = 0
      den = 0
      for index, entry in enumerate(df['description']):
        if df['general_classification'][index] == item:
          den+=1
          relevant = [df['ESCO_code'][index]]
          #Concatenate the title and the description
          entities = custom_pipeline(df['title'][index] + ' ' + entry)
          list_of_retrived_esco = []
          for entity in entities:
            if entity['type'] == "Occupation":
              list_of_retrived_esco.append(entity['retrieved'])
          if list_of_retrived_esco:
            #Calculate the Average Precision @K for the FIRST found Occupation against the relevant ESCO code.
            sum_of_ap1 += apk(relevant,list_of_retrived_esco[0],1)
            sum_of_ap5 += apk(relevant,list_of_retrived_esco[0],5)

      #Print for each class seperatly
      print(f"MAP@1 of {item} class is {sum_of_ap1/den}")
      sum_1 += sum_of_ap1
      print(f"MAP@5 of {item} class is {sum_of_ap5/den}")
      sum_2 += sum_of_ap5
      overall_den += den
    #Print MAP@1 (Accuracy) and MAP@5 for the whole evaluation dataset 
    print(f"MAP@1 {sum_1/overall_den}")
    print(f"MAP@5 {sum_2/overall_den}")
    

