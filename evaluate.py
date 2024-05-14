import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def plot_history(history, figsize=(16, 6)):
    '''Plots some interesting metrics from training history'''
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    
    axes[0].plot(history.history['loss'], label='train loss')
    axes[0].plot(history.history['val_loss'], label='val loss')
    axes[0].legend()
    axes[0].title.set_text('Train & Validation loss')
    
    axes[1].plot(history.history['accuracy'], label='train acc')
    axes[1].plot(history.history['val_accuracy'], label='val acc')
    axes[1].legend()
    axes[1].title.set_text('Train & Validation accuracy')

    plt.show()
    return fig


def evaluate_predictions(Y_true, Y_pred, genre_list):
    '''Evaluates predictions based on many metrics'''

    print(f'Accuracy: {accuracy_score(Y_true, Y_pred):0.2%}')
    print(f'Precision: {precision_score(Y_true, Y_pred, average="macro"):0.2%}')
    print(f'Recall: {recall_score(Y_true, Y_pred, average="macro"):0.2%}')
    print(f'F1: {f1_score(Y_true, Y_pred, average="macro")}')
    
    conf = confusion_matrix(Y_true, Y_pred, normalize='all')
    fig, ax = plt.subplots()
    ax.matshow(conf)
    
    genres = len(genre_list)
    ax.set_xticks(np.arange(genres), labels=genre_list)
    ax.set_yticks(np.arange(genres), labels=genre_list)
    
    for i in range(genres):
        for j in range(genres):
            text = ax.text(j, i, np.ceil(conf[i, j],3), ha="center", va="center", color="w", size=8)
    
    plt.title('Confusion Matrix for Classes')
    plt.show() 
    return fig 