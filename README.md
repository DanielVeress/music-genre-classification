# Music Genre Classification Based on Audio
### by Botond Rimmel & Dániel Veress

#

This is a project work for the Deep Learning course at [AIT](https://www.ait-budapest.com/).

This project uses the [gtzan](https://huggingface.co/datasets/marsyas/gtzan) dataset to train models and predict the genre of a given audio file. Our work consists of two main parts:

1. processing the data and extracting features for the model
2. create, train and evaluate the model

We train and test many different models, from simple Logistic Regression model to more complex CNN model.

---

## Setting up the environment

To download required modules using [*conda*](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) or *pip*:
```bash
    conda create --name <env_name> --file conda_requirements.txt
```
```bash
    pip install -r requirements.txt
```
The data is automatically loaded from [huggingface](https://huggingface.co/).

The data is processed in the DataProcessing.ipynb file, while the training and evaluation is done in hte ModelTraining.ipynb file.

NOTE: the SEGMENT_NUM variable sets the file used, so if the SEGMENT_NUM is wrong in the model training file, then an error could easily happen. Please watch out for this, and only use files that have been created with the DataProcessing.ipynb file before.

---

## Used Sources

[1] https://paperswithcode.com/task/music-genre-classification

[2] https://paperswithcode.com/task/music-classification