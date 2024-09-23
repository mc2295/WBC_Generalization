import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize

import pickle

dic_title = {'barcelona': 'Barcelona',
             'rabin' : 'Rabin', 
             'matek': 'Munich 2021', 
             'matek_2018': 'Munich 2018', 
             'lisc': 'LISC', 
             'jslh': 'JSLH',
             'jin_woo_choi': 'Jin Woo Choi', 
             'jiangxi_tecom': 'Jiangxi Tecom', 
             'bccd' : 'BCCD', 
             'tianjin_reviewed': 'Tianjin'}

def show_auc_per_class(labels, preds_tensor, list_labels_cat, source):
    
    precision = dict()
    recall = dict()
    average_precision = dict()
    set_classes = { i for i in labels}
    Y_test = label_binarize(labels, classes =list_labels_cat )
    n_classes = Y_test.shape[1]
    plt.figure()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], preds_tensor[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], preds_tensor[:, i])
        plt.plot(recall[i], precision[ i], lw=2, label=list_labels_cat[i])


    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(bbox_to_anchor=(1.01, 1.01))
    plt.title("Precision vs. Recall " + dic_title[source])
    plt.savefig('auc_' + source + '.png',  bbox_inches='tight')


def show_auc_all(labels, preds_tensor, list_labels_cat, source, datasets):
    precision = dict()
    recall = dict()
    labels = label_binarize(labels, classes =list_labels_cat )
    plt.figure()
    for k in source: 
        index = [i for i in range(len(datasets)) if datasets[i] == k]
        labels_dataset = np.stack([labels[i] for i in index])
        pred_tensor_dataset = np.stack([preds_tensor[i] for i in index])
        
        # precision[k], recall[k] = show_auc_per_class(labels_dataset, pred_tensor_dataset, list_labels_cat, k, savefig = False)
        precision[k], recall[k], _ = precision_recall_curve(labels_dataset.ravel(), pred_tensor_dataset.ravel())
        average_precision = average_precision_score(labels_dataset, pred_tensor_dataset, average='micro')
        auc_dataset = auc(recall[k], precision[ k])
        plt.plot(recall[k], precision[ k], lw=2, label= dic_title[k] + ' (AUC = {0:.2f})'.format(auc_dataset))
        # plt.plot(recall[k], precision[ k], lw=2, label= dic_title[k])
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(bbox_to_anchor=(1.01, 1.01))
    plt.title("Precision vs. Recall for each Dataset")
    plt.savefig('auc_all_sources_without_AP.png', bbox_inches='tight')  
        
