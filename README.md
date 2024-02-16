# Music Genre Classification

This is a project work for the Deep Learning course at [AIT](https://www.ait-budapest.com/).

The project's goal is to create a model, that is capable of predicting the genre of a given audio.

---

## Requirements

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

## Project Steps

1. Collect and preprocess the [dataset](https://www.tensorflow.org/datasets/catalog/gtzan), including audio file normalization, segmentation, and feature extraction using techniques such as Fourier transforms, mel frequency cepstral coefficients (MFCCs), and spectrograms.

2. Split the dataset into training, validation, and test sets.

MILESTONE 1

3. Design or adapt an LSTM, CNN and/or Transformer architecture for music genre classification.

4. Train the model  on the training set using an appropriate loss function and optimizer. First, use two music genres. Later, increase the number of genres and shorten the length of the analyzed music clips. 

MILESTONE 2

5. Tune hyperparameters such as learning rate, number of layers, and number of filters to improve performance on the validation set.

6. Evaluate the performance of the model on the test set using metrics such as accuracy, precision, recall, and F1 score.

7. Visualize the model's output to demonstrate its ability to accurately classify music by genre.

MILESTONE 3

## Evaluation


---

## Used Sources

https://paperswithcode.com/task/music-genre-classification
https://paperswithcode.com/task/music-classification