# tabiya-livelihoods-classifier
## Version 0

```python
from entity_linker import customPipeline
text = 'This a a dummy text :)'
custom_pipeline = customPipeline('path/to/your/CRF/model', 'all-MiniLM-L6-v2', crf=True)
extracted = custom_pipeline(text)
print(extracted)
```

## Train an Entity Extraction Model
Configure the necessary hyperparameters in the config.json file. Defaults are:

```javascript 
{
    "model_name":"bert-base-cased",
    "crf": false,
    "dataset_path":"job_ner_dataset",   
    "label_list" : [ "O","B-Skill","B-Qualification","I-Domain","I-Experience","I-Qualification","B-Occupation","B-Domain","I-Occupation","I-Skill","B-Experience"],
    "model_max_length" : 128,
    "batch_size" : 32,
    "learning_rate" : 1e-4,
    "epochs" : 4,
    "weight_decay": 0.01,
    "save":false,
    "output_path":"bert_job_ner"
}
```

Run the script

```
python train.py
```

## Bibiography 

A list on interesting and relevant matterial for reading:

* **Skills Entity Linking** Zhang, Mike, Rob van der Goot, and Barbara Plank. "Entity Linking in the Job Market Domain." arXiv preprint arXiv:2401.17979 (2024). 
* **Skills-ML** is an open-source Python library for developing and analyzing skills and competencies from unstructured text. (link: http://dataatwork.org/skills-ml/)   
* **SkillSpan**: Hard and Soft Skill Extraction from English Job Postings https://arxiv.org/abs/2204.12811 (Mike Zhang)
* **work2vec**: Using the full text of data from 200 million online job postings, we train and evaluate a natural language processing (NLP) model to learn the language of jobs. We analyze how jobs have changed in the past decade, and show how different words in the posting denote different occupations. We use this approach to create novel indexes of jobs, such as work-from-home ability. In ongoing work, we quantify the return to various skills. 
  
  https://digitaleconomy.stanford.edu/research/job2vec/
  https://digitaleconomy.stanford.edu/people/sarah-h-bana/
* **Data Science and ESCO** Insights into how ESCO is leveraging data-science techniques. https://esco.ec.europa.eu/en/about-esco/data-science-and-esco  
* **Machine Learning Assisted Mapping of Multilingual Occupational Data to ESCO**: A report that discusses the multilingual mapping
approach that the ESCO team established to support the maintenance of ESCO.  https://esco.ec.europa.eu/en/about-esco/publications/publication/machine-learning-assisted-mapping-multilingual-occupational
* **ESCO Publications**: Artificial intelligence & machine learning. https://esco.ec.europa.eu/en/about-esco/publications?f%5B0%5D=theme%3A109860&page=0  