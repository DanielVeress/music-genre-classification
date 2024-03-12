# Music Genre Classification Based on Audio
### by Boton Rimmel & Dániel Veress

#

This is a project work for the Deep Learning course at [AIT](https://www.ait-budapest.com/).

This project uses the [gtzan](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) dataset to train a model and predict the genre of a given audio file. Our work consists of 3 parts, first we process the data and create features for the model. In the second part, we train a model and optimize its parameters. Finally, we evaluate the model and try to optimize the model further.  

---

## Setting up the environment

To download required modules using [*conda*](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) or *pip*:
```bash
    conda create --name <env_name> --file requirements.txt
```
```bash
    pip install -r requirements.txt
```

To update requirements using *conda* or *pip*:
```bash
    conda list --export > requirements.txt
```
```bash
    pip freeze > requirements.txt
```

---

## Used Sources

[1] https://paperswithcode.com/task/music-genre-classification

[2] https://paperswithcode.com/task/music-classification