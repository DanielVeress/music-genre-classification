# Music Genre Classification Based on Audio
### by Botond Rimmel & Dániel Veress

#

This is a project work for the Deep Learning course at [AIT](https://www.ait-budapest.com/).

This project uses the [gtzan](https://huggingface.co/datasets/marsyas/gtzan) dataset to train a model and predict the genre of a given audio file. Our work consists of 3 parts:

1. processing the data and extracting features for the model
2. create, train and evaluate the model
3. enhancing the model (hyperparameter optimization and trying out different model types and techniques)

We expect the model to be able to predict the genres to an acceptable level. 

---

## Setting up the environment

To download required modules using [*conda*](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) or *pip*:
```bash
    conda create --name <env_name> --file requirements.txt
```
```bash
    pip install -r requirements.txt
```
The data is automatically loaded from [huggingface](https://huggingface.co/).

---

## Used Sources

[1] https://paperswithcode.com/task/music-genre-classification

[2] https://paperswithcode.com/task/music-classification