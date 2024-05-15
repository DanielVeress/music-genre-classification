import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def plot_history(history, figsize=(16, 6)):
    '''
    Plots some interesting metrics from training history
    
    Parameters:
    - history: training history
    - figsize: size of the plot

    Return:
    created figure
    '''
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    
    # loss plot
    axes[0].plot(history.history['loss'], label='train loss')
    axes[0].plot(history.history['val_loss'], label='val loss')
    axes[0].legend()
    axes[0].title.set_text('Train & Validation loss')
    
    # accuracy plo 
    axes[1].plot(history.history['accuracy'], label='train acc')
    axes[1].plot(history.history['val_accuracy'], label='val acc')
    axes[1].legend()
    axes[1].title.set_text('Train & Validation accuracy')

    plt.show()
    return fig


def evaluate_predictions(Y_true, Y_pred, genre_list, normalize=None, text_size=10):
    '''
    Evaluates predictions based on many metrics
    
    Parameters:
    - Y_true: true labels
    - Y_pred: predicted labels
    - genre_list: a list with the unique genres
    - normalize: normalization used for the confusion matrix, {'true', 'pred', 'all'}, default=None
    - text_size: text size in the confusion matrix plot

    Returns:
    created figure
    '''

    # general metrics
    print(f'Accuracy: {accuracy_score(Y_true, Y_pred):0.2%}')
    print(f'Precision: {precision_score(Y_true, Y_pred, average="macro"):0.2%}')
    print(f'Recall: {recall_score(Y_true, Y_pred, average="macro"):0.2%}')
    print(f'F1: {f1_score(Y_true, Y_pred, average="macro")}')
    
    # confusion matrix
    conf = confusion_matrix(Y_true, Y_pred, normalize=normalize)
    genres = len(genre_list)
    fig, ax = plt.subplots()
    ax.matshow(conf)
    ax.set_xticks(np.arange(genres), labels=genre_list)
    ax.set_yticks(np.arange(genres), labels=genre_list)
    # setting labels on the plot
    for i in range(genres):
        for j in range(genres):
            text = ax.text(j, i, conf[i, j], ha="center", va="center", color="w", size=text_size)
    plt.title('Confusion Matrix for Classes')
    plt.show() 
    return fig 