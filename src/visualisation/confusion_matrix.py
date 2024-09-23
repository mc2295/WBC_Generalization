from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from fastai.vision.all import itertools


def plot_confusion_matrix(cm, cm_norm, list_labels_cat):
    cmap = "Blues"
    title = 'Confusion matrix'
    norm_dec:int=2  # Decimal places for normalized occurrences
    plot_txt:bool=True # Display occurrence in matrix
    fig = plt.figure()
    plt.imshow(cm_norm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize = 20)
    tick_marks = np.arange(len(list_labels_cat))
    plt.xticks(tick_marks, list_labels_cat, rotation=90, fontsize=20)
    plt.yticks(tick_marks, list_labels_cat, rotation=0, fontsize=20)

    if plot_txt:
        thresh = cm_norm.max() / 2.
        for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
            coeff_norm = f'{cm_norm[i, j]:.{norm_dec}f}'
            coeff = f'{cm[i, j]}'
            plt.text(j, i, coeff_norm, horizontalalignment="center", verticalalignment="center",
                     color="white" if cm_norm[i, j] > thresh else "black", fontsize=20)
            plt.text(j, i+0.3, coeff, horizontalalignment="center", verticalalignment="center",
                     color="white" if cm_norm[i, j] > thresh else "black", fontsize=15)

    ax = fig.gca()
    ax.set_ylim(len(list_labels_cat)-.5,-.5)

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('report/confusion_matrix.png')
    plt.show(block=False)


def show_confusion_matrix(pred_label, real_label, list_labels, plot_matrix):
    cm = confusion_matrix(real_label, pred_label[:len(real_label)],  labels = list_labels)
    
    cm_norm_recall = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm_precision = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]# axis = 0 pour precision 
    cm_without_nan = np.nan_to_num(cm)
    acc = cm_without_nan.diagonal().sum()/cm_without_nan.sum()
    recall_per_class = cm.diagonal()/cm_without_nan.sum(axis=1)
    precision_per_class = cm.diagonal()/cm_without_nan.sum(axis = 0)
    # Round the values to 3 decimal places
    cm = np.round(cm, 3)
    cm_norm_recall = np.round(cm_norm_recall, 3)
    acc = round(acc, 3)
    precision_per_class = np.round(precision_per_class, 3)
    recall_per_class = np.round(recall_per_class, 3)
    if plot_matrix:
        plot_confusion_matrix(cm, cm_norm_recall, list_labels)
    return recall_per_class, precision_per_class, acc
