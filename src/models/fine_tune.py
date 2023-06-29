from sklearn.model_selection import StratifiedKFold
from fastai.vision.all import ClassificationInterpretation, Learner, accuracy, LabelSmoothingCrossEntropy, TrackerCallback
from data.create_variables import create_df, create_splits, create_dataloader_single_image
import torch
from models.model_params import split_layers
import numpy as np
import csv
import os
import optuna
from optuna.integration import FastAIPruningCallback


# class FastAIPruningCallback(TrackerCallback):
#     def __init__(self, learn, trial, monitor):
#         # type: (Learner, optuna.trial.Trial, str) -> None

#         super(FastAIPruningCallback, self).__init__(learn, monitor)

#         self.trial = trial

#     def on_epoch_end(self, epoch, **kwargs):
#         # type: (int, Any) -> None

#         value = self.get_monitor_value()
#         if value is None:
#             return

#         self.trial.report(value, step=epoch)
#         if self.trial.should_prune():
#             message = 'Trial was pruned at epoch {}.'.format(epoch)
#             raise optuna.structs.TrialPruned(message)


class optimize_fine_tune():
    def __init__(self, entry_path, source, fine_tune_size, balanced, files_to_look_at, transform, preds_to_look_at, list_labels_cat, batchsize, model_name, file_report_name, head_fine_tune, k_fold = 3):
        self.skf = StratifiedKFold(n_splits = k_fold, shuffle = True, random_state = 1)
        self.entry_path = entry_path
        self.source = source
        self.fine_tune_size = fine_tune_size
        self.balanced = balanced
        self.head_fine_tune = head_fine_tune
        self.transform = transform
        self.preds_to_look_at = preds_to_look_at
        self.files_to_look_at = files_to_look_at
        self.df = create_df(self.entry_path + 'references', self.source, list_labels_cat)
        self.splits = create_splits(self.entry_path, self.source, self.df, self.fine_tune_size, balanced = self.balanced, files_to_look_at = self.files_to_look_at)
        self.df_reduced = self.df.iloc[self.splits[0]]
        self.file_report_name =  file_report_name
        self.batchsize = batchsize
        self.model_name = model_name

    def objective(self, trial):

        # Generate the optimizers.
        lr1 = trial.suggest_float("lr", 1e-7, 1e-6, log=True)
        # lr2 = trial.suggest_float("lr", 1e-2, 1e-1, log=True)
        lr2 = 1e-2
        lr = [lr1, lr2]
        epoch_nb = trial.suggest_int('epoch_nb', 10, 35)

        acc_list = []
        acc_per_class_list = []
        loss_list = []
        # Training of the model.

        for k, (train_index, val_index) in enumerate(self.skf.split(self.df_reduced.index, self.df_reduced.label)):

            model = torch.load(self.entry_path + 'models/Mixed_sources_trained/Barcelona_Saint_Antoine/resnet/' + self.model_name)

            dls = create_dataloader_single_image(self.entry_path, self.batchsize, self.df_reduced,[train_index.tolist(), val_index.tolist()] , transform = self.transform)
            learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(), metrics = accuracy, splitter = split_layers)
            if self.head_fine_tune:
                learn.freeze()
            callback = FastAIPruningCallback(trial, monitor='valid_loss')
            for epoch in range(epoch_nb):
                learn.fit_one_cycle(1, lr, cbs=callback)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            loss, acc = learn.validate()
            interp = ClassificationInterpretation.from_learner(learn)
            cm = interp.confusion_matrix()
            acc_per_class = cm.diagonal()/cm.sum(axis=1)

            loss_list.append(loss)
            acc_list.append(acc)
            acc_per_class_list.append(acc_per_class)

        acc_tot = np.mean(np.array(acc_list))
        acc_per_class_tot = np.mean(np.array(acc_per_class_list), axis = 0)

        loss_tot = np.mean(np.array(loss_list))
        with open(self.file_report_name, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([self.model_name, ', '.join(self.source), self.fine_tune_size, epoch_nb, self.batchsize, lr, self.head_fine_tune, self.balanced, self.preds_to_look_at, 'avg cv', acc_tot] + list(acc_per_class_tot))

        return loss_tot
