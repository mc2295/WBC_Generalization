from sklearn.model_selection import StratifiedKFold
from fastai.vision.all import ClassificationInterpretation, Learner, accuracy, LabelSmoothingCrossEntropy, TrackerCallback
from data.create_variables import create_df, create_splits, create_dataloader_single_image
import torch
from models.model_params import split_layers
import pickle
import numpy as np
import csv
import os
import optuna
from optuna.integration import FastAIPruningCallback

def fine_tune_CV(entry_path, df_filename, model_name, lr, epoch_nb, batchsize, transform, balanced, optim_fine_tune = False, trial = None):
    file = open(df_filename[0], 'rb')
    df = pickle.load(file)
    df_reduced = df.loc[df['is_valid'] == False]

    skf = StratifiedKFold(n_splits = 3, shuffle = True, random_state = 1)
    acc_list = []
    acc_per_class_list = []
    loss_list = []
    for k, (train_index, val_index) in enumerate(skf.split(df_reduced.index, df_reduced.label)):
        model = torch.load(entry_path + 'models/Mixed_sources_trained/Barcelona_Saint_Antoine/' + model_name)
        dls = create_dataloader_single_image(entry_path, batchsize, df_reduced,[train_index.tolist(), val_index.tolist()] , transform = transform)
        learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(), metrics = accuracy, splitter = split_layers)
        if optim_fine_tune:
            callback = FastAIPruningCallback(trial, monitor='valid_loss')
            for epoch in range(epoch_nb):
                learn.fit_one_cycle(1, lr, cbs=callback)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        else:
            learn.fit_one_cycle(epoch_nb, lr)
        loss, acc = learn.validate()
        interp = ClassificationInterpretation.from_learner(learn)
        cm = interp.confusion_matrix()
        cm_without_nan = np.nan_to_num(cm)
        acc_per_class = cm.diagonal()/cm_without_nan.sum(axis=1)

        loss_list.append(loss)
        acc_list.append(acc)
        acc_per_class_list.append(acc_per_class)

    acc_tot = np.mean(np.array(acc_list))
    acc_per_class_tot = np.mean(np.array(acc_per_class_list), axis = 0)

    loss_tot = np.mean(np.array(loss_list))
    return acc_tot, acc_per_class_tot, loss_tot, learn.model

class optimize_fine_tune():
    def __init__(self, entry_path, source, balanced, head_fine_tune, transform, df_filename, file_report_name, model_name, preds_to_look_at):
        self.entry_path = entry_path
        self.source = source
        self.fine_tune_size = 100
        self.balanced = balanced
        self.head_fine_tune = head_fine_tune
        self.transform = transform
        self.df_filename = df_filename
        self.file_report_name =  file_report_name
        self.model_name = model_name
        self.preds_to_look_at = preds_to_look_at

    def objective(self, trial):
        lr = trial.suggest_int("lr", 10, 1000, log=True)*1e-6
        batchsize = trial.suggest_int("batchsize", 8, 20)
        epoch_nb = trial.suggest_int('epoch_nb', 15, 50, 5)
        acc_tot, acc_per_class_tot, loss_tot, model = fine_tune_CV(self.entry_path, self.df_filename, self.model_name, lr, epoch_nb, batchsize, self.transform, self.balanced, optim_fine_tune = True, trial = trial)
        with open(self.file_report_name, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([self.model_name, ', '.join(self.source), self.fine_tune_size, epoch_nb, batchsize, lr, self.head_fine_tune, self.balanced, self.preds_to_look_at, 'avg cv', acc_tot] + list(acc_per_class_tot))

        return loss_tot
