import torch
import matplotlib.pyplot as plt
import numpy as np
import optuna
from fastai.vision.all import create_head, accuracy, ToTensor, Learner, LabelSmoothingCrossEntropy
# from torch.nn import BCELoss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import sklearn.metrics as metrics
from data.create_variables import create_dls, list_labels_cat
from models.create_model import create_model, create_learner
from models.encoder_head_model import Model
from models.model_params import split_layers
from visualisation.create_plots import scatter_plot_embeddings, create_df_of_2D_embeddings_info, visualise_images_in_region, plot_distribution, show_images_with_preds, show_histo_of_pred_tensor, show_batch, plot_metrics
from visualisation.create_embeddings import flatten_batch, create_matrix_of_flattened_images, create_embeddings, create_preds, show_confusion_matrix, evaluate_cluster, get_bootstrap_confidence_interval, save_preds
from models.fine_tune import optimize_fine_tune, fine_tune_CV
import csv
import os

class Workflow():
    def __init__(self, entry_path):
        self.entry_path = entry_path
        self.source = ['barcelona', 'matek_2018']
        self.size = [1000,1000]
        self.batchsize = 16
        self.transform = True
        self.epoch_nb = 100
        self.lr = 1e-5
        self.nb_of_iterations = 4
        self.model = None
        self.head_fine_tune = False
        self.preds_to_look_at = None
        self.balanced = True
        self.files_to_look_at = None
        self.plot_losses = True
        self.test_set = True
        self.df_filename = None
        self.model_name = 'siamese8_transform3_classifier_head'
        self.ML_classifier = 'KNN'

    def find_learning_rate(self):
        dls = create_dls(self.entry_path, self.source, workflow.model.siamese_head, batchsize = self.batchsize, size = self.size, transform = self.transform, df_filename = self.df_filename)
        learn = create_learner(self.model, dls, weights = None)
        lr = learn.lr_find()
        print(lr)
        return lr

    def show_batch(self, siamese_head = False):
        dls = create_dls(self.entry_path, self.source,siamese_head, batchsize = self.batchsize, size = self.size, transform = self.transform, df_filename = self.df_filename)
        b = show_batch(dls, siamese_head = siamese_head)
        return b

    def train_model(self, siamese_head = False, weights = False, architecture = 'efficientnet'):
        model = create_model(siamese_head, architecture)
        dls = create_dls(self.entry_path, self.source, model.siamese_head, batchsize = self.batchsize, size = self.size, transform = self.transform, df_filename = self.df_filename)
        learn = create_learner(model, dls, weights = weights)
        learn.fit_one_cycle(self.epoch_nb, self.lr)
        if self.plot_losses:
            learn.recorder.plot_metrics()
        return learn.model

    def fine_tune(self, with_CV = False):
        if self.preds_to_look_at is not None:
            dls_0 = create_dls(self.entry_path, self.source, self.model.siamese_head, batchsize = self.batchsize, size = [10000], transform = self.transform,balanced = False, df_filename = self.df_filename)
            _, _, _, _, _, files_to_look_at = create_preds(dls_0, self.model, test_set = False, preds_to_look_at = self.preds_to_look_at)
            self.files_to_look_at = files_to_look_at
        elif with_CV:
            # cross validation sur df_reduced
            acc_tot, acc_per_class_tot, loss_tot, model = fine_tune_CV(self.entry_path, self.df_filename, self.model_name, self.lr, self.epoch_nb, self.batchsize, self.transform, self.balanced)
            return acc_per_class_tot, acc_tot

        else:
            dls = create_dls(self.entry_path, self.source, self.model.siamese_head, self.batchsize, size=self.size, transform = self.transform, balanced = self.balanced, df_filename = self.df_filename)
            learn = create_learner(self.model, dls, weights = None)
            if self.head_fine_tune:
                learn.freeze()
            learn.fit_one_cycle(self.epoch_nb, self.lr)
            return learn.model

    def optimize_fine_tune(self, file_report_name):
        optim = optimize_fine_tune(self.entry_path, self.source, self.balanced, self.head_fine_tune, self.transform, self.df_filename, file_report_name, self.model_name, self.preds_to_look_at)
        study = optuna.create_study()
        study.optimize(optim.objective)
        best_trial = study.best_trial
        return best_trial

    def show_confusion_matrix(self, plot_matrix = False):

        dls = create_dls(self.entry_path, self.source, self.model.siamese_head, batchsize = self.batchsize, size=self.size, transform = self.transform, df_filename = self.df_filename, full_evaluation = True)
        print(len(dls.train_ds), len(dls.valid_ds))
        y = create_preds(dls, self.model, test_set = True)
        if self.model.siamese_head:
            preds, targs = y
        else:
            _, preds, _, targs, _, _ = y
        recall_per_class, precision_per_class, acc = show_confusion_matrix(preds, targs, dls.vocab[0], plot_matrix)
        return recall_per_class, precision_per_class, acc

    def show_flattened_images_in_2D(self, image_nb_per_source = 32):
        X, dataset= create_matrix_of_flattened_images(self.entry_path, image_nb_per_source, self.source, transform = self.transform)
        labels = []
        data = scatter_plot_embeddings(X, labels, 't-SNE', dataset, None, display_classes = False)
        return data

    def show_embeddings_2D(self, method = 't-SNE', display_classes = True):
        dls = create_dls(self.entry_path, self.source, False, batchsize = self.batchsize, size = self.size, transform = self.transform, balanced = self.balanced, df_filename = self.df_filename)
        embedding, labels, dataset, filenames = create_embeddings(self.model, dls, test_set = self.test_set)
        df_embeddings = scatter_plot_embeddings(embedding, labels, method, dataset, filenames, display_classes)
        return df_embeddings

    def show_images_from_scatter_plot_region(self, data, cell_class, x_region, y_region):
        group = visualise_images_in_region(self.entry_path, data, cell_class, x_region, y_region)
        return group

    def show_distribution_of_prediction(self):
        dls = create_dls(self.entry_path, self.source, self.model.siamese_head, batchsize = self.batchsize, size=self.size, transform = self.transform, df_filename = self.df_filename)
        preds_tensor, preds_label, preds_proba, labels, dataset, filenames = create_preds(dls, self.model, test_set = True)
        plt.figure()
        plot_distribution(preds_label, title = 'Predicted Labels')
        plt.figure()
        plot_distribution(labels, title = 'Actual Labels')

    def show_predictions_images(self):
        dls = create_dls(self.entry_path, self.source, self.model.siamese_head, batchsize = self.batchsize, size=self.size, transform = self.transform, df_filename = self.df_filename)
        preds_tensor, preds_label, preds_proba, labels, dataset, filenames = create_preds(dls, self.model, test_set = True)
        plt.figure()
        show_images_with_preds(preds_label, preds_proba, labels, filenames)
        plt.figure()
        show_histo_of_pred_tensor(preds_tensor)

    def report_results_accuracy(self):
        acc_per_class = np.zeros((self.nb_of_iterations, 6))
        acc =  []

        if not os.path.exists('fine_tune_results.csv'):
            with open('fine_tune_results.csv', 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['model', 'test_set', 'fine_tune size', 'epoch_nb','lr', 'head_fine_tune', 'balanced', 'preds_to_look_at', 'iteration', 'average_acc', 'basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'neutrophil', 'monocyte'])

        for i in [32, 50, 75, 100]:
            for k in range(self.nb_of_iterations):
                self.size = [i]
                self.source = ['matek'] # pour fine tune
                self.model = torch.load(self.entry_path + 'models/Mixed_sources_trained/Barcelona_Saint_Antoine/' + self.model_name)
                model = self.fine_tune()
                self.size = [5000]
                self.model = model
                self.source = ['matek'] # pour évaluer
                res = self.show_confusion_matrix(plot_matrix = False)

                with open('fine_tune_results.csv', 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.model_name, ', '.join(self.source), i,self.epoch_nb, self.lr, self.head_fine_tune, self.balanced, self.preds_to_look_at, k+1, res[1]] + res[0].tolist())

    def report_results_clusters(self, model_name, file_report_name, source, transform, size, test_set = True, batchsize = 16, preds_to_look_at = None, balanced=None, show_graph = False, fine_tune = False, lr = None, fine_tune_size = None, epoch_nb = None):
        model = torch.load(self.entry_path + 'models/Mixed_sources_trained/Barcelona_Saint_Antoine/resnet/' + model_name)
        if fine_tune:
            if isinstance(lr, list):
                head_fine_tune = False
            else:
                head_fine_tune = True
            model = self.fine_tune( model, source, batchsize, epoch_nb, lr, size = fine_tune_size, transform = transform, head_fine_tune = head_fine_tune, balanced = balanced, preds_to_look_at = preds_to_look_at)
        size_str = [str(i) for i in size]
        dls = create_dls(self.entry_path, source, False, batchsize = batchsize, size = size, transform = transform, balanced = True)
        embedding, labels, dataset, filenames = create_embeddings(model, dls, test_set = test_set)
        if show_graph:
            plt.figure()
            df_embeddings = scatter_plot_embeddings(embedding, labels, 't-SNE', dataset, filenames, display_classes = True)
        silhouette, bouldin, calinski, average_intra_cluster_distance, average_inter_cluster_distance, silhouette_2D, bouldin_2D, calinski_2D, average_intra_cluster_distance_2D, average_inter_cluster_distance_2D = evaluate_cluster(embedding, labels, size)
        if not os.path.exists(file_report_name):
            with open(file_report_name, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['model', 'test_set', 'nb of images', 'fine_tune', 'fine_tune_size', 'lr','epoch_nb','preds_to_look_at', 'balanced','transform','batchsize','silhouette score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index', 'Intra-Cluster Distance', 'Inter-Cluster Distance', 'silhouette score_2D', 'Davies-Bouldin Index_2D', 'Calinski-Harabasz Index_2D', 'Intra-Cluster Distance_2D', 'Inter-Cluster Distance_2D'])

        with open(file_report_name, 'a') as file:
            writer = csv.writer(file)
            writer.writerow([model_name, ', '.join(source), ', '.join(size_str), fine_tune, fine_tune_size, lr, epoch_nb, preds_to_look_at, balanced, transform, batchsize, silhouette, bouldin, calinski, average_intra_cluster_distance, average_inter_cluster_distance, silhouette_2D, bouldin_2D, calinski_2D, average_intra_cluster_distance_2D, average_inter_cluster_distance_2D])

    def bootstrap(self):
        dls = create_dls(self.entry_path, self.source, self.model.siamese_head, batchsize = self.batchsize, size=self.size, transform = self.transform, df_filename = self.df_filename, full_evaluation = True)
        y = create_preds(dls, self.model, test_set = True)
        _, preds, _, targs, _, _ = y
        save_preds(self, targs, preds)
        r = get_bootstrap_confidence_interval(targs, preds, self.source[0], dls.vocab[0], draw_number=1000)
        return r
