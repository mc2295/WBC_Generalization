import pandas as pd
import numpy as np
import copy
from visualisation.confusion_matrix import show_confusion_matrix

def save_preds(targs, preds, lr, batchsize, source):
    df = pd.DataFrame(columns=["size_evaluation", "batchsize", "lr", "targs", "preds"])
    result = {"size_evaluation" : len(preds), "batchsize" : batchsize, "lr" : lr, "targs" : targs, "preds" : preds}
    df.loc[len(df)] = result
    df.to_csv('report/prediction/preds_' + source[0] + '.csv')



def get_bootstrap_confidence_interval(y_true, y_pred, source, vocab,
                                      draw_number=1000, alpha=5.0, return_outputs=False):
    y_true_copy = [copy.deepcopy(y_true)]
    y_pred_copy = [copy.deepcopy(y_pred)]
    
    recall_per_class_tot = []
    precision_per_class_tot = []
    acc_tot = []
    main_results = []
    # For each result to evaluate
    for sublist_true, sublist_pred in zip(y_true_copy, y_pred_copy): 
        np_y_true = np.asarray(sublist_true)
        np_y_pred = np.asarray(sublist_pred)
        # main_results.append([m(sublist_true, sublist_pred) for m in metric_function_list])

        # Perform `draw_number` draw with replacement of the sample
        # and compute the metrics at each time
        i = 0
        while i < draw_number: 
            # random draw with replacement
            random_indices = np.random.randint(low=0, high=len(sublist_true), size=len(sublist_true))
            
            this_y_pred = np_y_pred[random_indices]
            this_y_true = np_y_true[random_indices]
            # skip if no positive label is present
            if not any([e != 0 for e in this_y_true]):
                continue
            
            # get metrics for this random sample
            i += 1
            recall_per_class, precision_per_class, acc = show_confusion_matrix(this_y_pred, this_y_true, vocab, False)
            recall_per_class_tot.append(recall_per_class)
            precision_per_class_tot.append(precision_per_class)
            acc_tot.append([acc])
    df = pd.DataFrame(columns=["metric", "class","mean", "median", "lower_p", "upper_p"])
    metric = ["acc", "precision_per_class", "recall_per_class"]
    for i, met in enumerate([acc_tot, precision_per_class_tot, recall_per_class_tot]):
        np_acc = np.array(met)
        # calculate 95% confidence intervals (1 - alpha)
        lower_p = alpha / 2
        lower = np.maximum(0, np.percentile(np_acc, lower_p, axis=0))
        upper_p = (100 - alpha) + (alpha / 2)
        upper = np.minimum(1, np.percentile(np_acc, upper_p, axis=0))
        medians = np.median(np_acc, axis=0)
        means = np.mean(np_acc, axis=0)
        mains = np.mean(np_acc, axis=0)
    
        if i == 0:
            result = {"metric": metric[i],"class" : "","mean":means[0], "median":medians[0], "lower_p":lower[0], "upper_p":upper[0]} 
            df.loc[len(df)] = result
        else: 
            for k in range(len(means)):
                result = {"metric": metric[i], "class" : vocab[k], "mean":means[k], "median":medians[k], "lower_p":lower[k], "upper_p":upper[k]}
                df.loc[len(df)] = result
    df.to_csv('report/incertitudes/incertitudes_' + source+ '.csv')
        
    return result