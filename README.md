# Tabiya Livelihoods Classifier

The Tabiya Livelihoods Classifier provides an easy-to-use implementation of the entity-linking paradigm to support job description heuristics. Using state-of-the-art transformer neural networks, this tool can extract five entity types: Occupation, Skill, Qualification, Experience, and Domain. For the Occupations and Skills, ESCO-related entries are retrieved. The procedure consists of two discrete steps: entity extraction and similarity vector search.

## Table of Contents

- **[Use the model](inference/README.md)**: Instructions on how to use the inference tool.
- **[Training](train/README.md)**: Details on how to train the model.
- **[Model's Architecture](#models-architecture)**
- **[Installation](#installation)**
- **[License](#license)**
- **[Bibliography](#bibliography)**

## Model's Architecture

![Model Architecture](./pics/entity_linker.png)

## Installation

### Prerequisites

- A recent version of [git](https://git-scm.com/) (e.g. ^2.37 )
- [Python 3.10 or higher](https://www.python.org/downloads/)
- [Poerty 1.8 or higher](https://python-poetry.org/)
  > Note: to install Poetry consult the [Poetry documentation](https://python-poetry.org/docs/#installing-with-the-official-installer) 
- [Git LFS](https://git-lfs.github.com/)

### Using Git LFS

This repository uses Git LFS for handling large files. Before you can use this repository, you need to install and set up Git LFS on your local machine.
See https://git-lfs.com/ for installation instructions.

After Git LFS is set up, follow these steps to clone the repository:

```shell
git clone https://github.com/tabiya-tech/tabiya-livelihoods-classifier.git
```

If you already cloned the repository without Git LFS, run:

```shell
git lfs pull
```

#### Set up virtualenv
In the **root directory** of the backend project (so, the same directory as this README file), run the following commands:

```shell
# create a virtual environment
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate
```

### Install the dependencies

```shell
# Use the version of the dependencies specified in the lock file
poetry lock --no-update
# Install missing and remove unreferenced packages
poetry install --sync
```

> Note: Install poetry system-wide (not in a virtualenv).

> Note:
> Install the dependencies for the training using:
>  ```shell
> # Use the version of the dependencies specified in the lock file
> poetry lock --no-update
> # Install missing and remove unreferenced packages
> poetry install --sync --with train
>  ```

> Note:
> Before running any tasks, activate the virtual
> environment so that the installed dependencies are available:
>  ```shell
>  # activate the virtual environment
>  source venv/bin/activate
>  ```
> To deactivate the virtual environment, run:
> ```shell
> # deactivate the virtual environment
> deactivate
> ```

Activate Python and download the NLTK punctuation package to use the sentence tokenizer. You only need to download `punkt` once.

```shell
python
import nltk
nltk.download('punkt')
```
## License

The code and model weights are licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

The dataset is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0). See the [DATA_LICENSE](./DATA_LICENSE) file for details.

## Bibiography 

A list on interesting and relevant matterial for reading:
* **GPT NER** [GPT-NER: Named Entity Recognition via Large Language Models](https://arxiv.org/pdf/2304.10428) (Shuhe Wang)
* **Skill Extraction with LLMs** [Rethinking Skill Extraction in the Job Market Domain using Large Language Models](https://arxiv.org/pdf/2402.03832) (Mike Zhang)
* **NER annotation with LLM** [LLMs Accelerate Annotation for Medical Information Extraction](https://proceedings.mlr.press/v225/goel23a)
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
