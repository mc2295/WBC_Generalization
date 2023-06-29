import torch
import matplotlib.pyplot as plt
import numpy as np
import optuna
from fastai.vision.all import create_head, accuracy, ToTensor, Learner, LabelSmoothingCrossEntropy
from models.siamese.siamese_params import BCE_loss, siamese_splitter, my_accuracy
# from torch.nn import BCELoss
from sklearn.neighbors import KNeighborsClassifier
from data.siamese_image import SiameseTransform, show_batch
from data.create_variables import create_dls, list_labels_cat
from models.create_model import create_model, create_learner
from models.encoder_head_model import Model
from models.model_params import split_layers
from visualisation.create_plots import scatter_plot_embeddings, create_df_of_2D_embeddings_info, visualise_images_in_region, plot_distribution, show_images_with_preds, show_histo_of_pred_tensor
from visualisation.KNN import plot_correlation, KNN_weighted
from visualisation.create_embeddings import flatten_batch, create_matrix_of_flattened_images, create_embeddings, create_preds, show_confusion_matrix, evaluate_cluster
from visualisation.label_with_siamese import label_files
from models.fine_tune import optimize_fine_tune
import csv
import os

# from nbdev import show_doc

class Workflow():
    def __init__(self, entry_path):
        self.entry_path = entry_path

    def find_learning_rate(self, model, source, batchsize = 32, size = [0,0,0,0], transform = False):
        '''
        find the optimal learning rate for training a model
        '''
        dls = create_dls(self.entry_path, source, model.siamese_head, batchsize = batchsize, size = size, transform = transform)
        learn = create_learner(model, dls, weights = None)
        learn.lr_find()
        print(learn.lr_find())
        return learn.lr_find()

    def show_batch(self, source, transform = False, siamese_head = False, size = [0,0,0,0]):
        '''
        show images of a batch, with a source dataset
        '''
        dls = create_dls(self.entry_path, source, siamese_head=siamese_head, transform=transform, size = size) # dataloader
        if siamese_head:
            b = dls.one_batch()
            show_batch(b, None, None)
        else:
            b = dls.one_batch()[0] # [0] to have the images, [1] to have the labels
            fig, axes = plt.subplots(5,5, figsize = ((10,10)))
            for i, ax in enumerate(axes.flat):
                ax.imshow(b[i].permute(2,1,0).to('cpu'))
                ax.axis('off')
        return b

    def train_model(self, source_train, epoch_nb, siamese_head,  lr, batchsize = 32, size = [0,0,0,0], transform = False, weights = False):
        '''
        input:
        source_train : datasets,
        siamese_head: True or False if we want a siamese network,
        lr : learning rate
        size : nb of images per dataset,
        transform : if we want to transform images,
        weights: if we want to balance batches by oversampling
        result: model trained from scratch on source_train dataset
        '''
        model = create_model(siamese_head)
        dls = create_dls(self.entry_path, source, model.siamese_head, batchsize = batchsize, size = size, transform = transform)
        learn = create_learner(model, dls, weights = None)
        learn.fit_one_cycle(epoch_nb, lr)
        return learn.model

    def fine_tune(self, model, source_tuning, batchsize, epoch_nb, lr, size = [0,0,0,0], transform = False, head_fine_tune = True, balanced = False, preds_to_look_at = None):
        '''
        input:
        source_tuning: dataset with fine tuning images
        lr: learning rate
        size: the number of images per dataset for fine tune
        head_fine_tune: True only head is fine tune or false, head and body are fine tuned
        balanced: if we want balanced classes in the fine tune subset
        preds_to_look_at: if we want to take images among best predictions or worst predictions for fine tuning, preds_to_look_at can take the values 'best_preds' or 'worst_preds'
        results: model after fine tuning
        '''
        if preds_to_look_at is not None:
            dls_0 = create_dls(self.entry_path, source_tuning, model.siamese_head, batchsize = batchsize, size = [10000], transform = transform, files_to_look_at = None, balanced = balanced)
            _, _, _, _, _, files_to_look_at = create_preds(dls_0, model, test_set = False, preds_to_look_at = preds_to_look_at)
        else:
            files_to_look_at = None
        dls = create_dls(self.entry_path, source_tuning, model.siamese_head, batchsize = batchsize, size = size, transform = transform, files_to_look_at = files_to_look_at)
        learn = create_learner(model, dls, weights = None)
        if head_fine_tune:
            learn.freeze()
        learn.fit_one_cycle(epoch_nb, lr)
        return learn.model

    def optimize_fine_tune(self, source, fine_tune_size, balanced, files_to_look_at, transform, preds_to_look_at, list_labels_cat, batchsize, model_name, file_report_name, head_fine_tune, k_fold = 3):
        optim = optimize_fine_tune(self.entry_path, source, fine_tune_size, balanced, files_to_look_at, transform, preds_to_look_at, list_labels_cat, batchsize, model_name, file_report_name, head_fine_tune, k_fold = k_fold)
        study = optuna.create_study()
        study.optimize(optim.objective)
        best_trial = study.best_trial
        return best_trial

    def show_confusion_matrix(self, model, source_test, batchsize = 32, size = [0,0,0,0], transform = False, plot_matrix = False, normalize = True):
        dls = create_dls(self.entry_path, source_test, model.siamese_head, batchsize = batchsize, size=size, transform = transform)
        y = create_preds(dls, model, test_set = True)
        if model.siamese_head:
            preds, targs = y
        else:
            _, preds, _, targs, _, _ = y
        acc_per_class, acc = show_confusion_matrix(preds, targs, dls.vocab[0], plot_matrix,  normalize)
        return acc_per_class, acc

    def show_flattened_images_in_2D(self, image_nb_per_source, source, transform = False):
        '''
        to visualise images in space from different dataset. We flatten images and project them in 2D.
        '''
        X, dataset= create_matrix_of_flattened_images(self.entry_path, image_nb_per_source, source, transform = transform)
        labels = []
        data = scatter_plot_embeddings(X, labels, 't-SNE', dataset, None, display_classes = False)
        return data

    def show_embeddings_2D(self, model, source, method, display_classes = True, batchsize = 32,test_set = False, transform = False, size = [0,0,0,0]):
        '''
        to plot the embeddings of the last layer of the encoder in 2D. We should see clusters by classes.
        '''
        dls = create_dls(self.entry_path, source, False, batchsize = batchsize, size = size, transform = transform, balanced = True)
        embedding, labels, dataset, filenames = create_embeddings(model, dls, test_set = test_set)
        df_embeddings = scatter_plot_embeddings(embedding, labels, method, dataset, filenames, display_classes)
        return df_embeddings

    def show_images_from_scatter_plot_region(self, data, cell_class, x_region, y_region, transform = False):
        '''
        when we want to see the images associated to the points of the clusters produced by the previous function.
        '''
        group = visualise_images_in_region(self.entry_path, data, cell_class, x_region, y_region)
        return group

    def show_distribution_of_prediction(self, model, source, batchsize = 32, test_set = True, transform = True, size = [0,0,0,0]):
        '''
        to show the distribution of classes predicted by the model, and the distribution of the real labels.
        '''
        dls = create_dls(self.entry_path, source, model.siamese_head, batchsize = batchsize, size=size, transform = transform)
        preds_tensor, preds_label, preds_proba, labels, dataset, filenames = create_preds(dls, model, test_set = True)
        plt.figure()
        plot_distribution(preds_label, title = 'Predicted Labels')
        plt.figure()
        plot_distribution(labels, title = 'Actual Labels')

    def show_predictions_images(self,  model, source, batchsize = 32, test_set = True, transform = True, size = [0,0,0,0], preds_to_look_at = None):
        '''
        to show images, with their predicted labels, real label and the probability of having this label. We can show only best predictions
        or worst predictions by walling preds_to_look_at = "best_preds" or "worst_preds"
        '''
        dls = create_dls(self.entry_path, source, model.siamese_head, batchsize = batchsize, size=size, transform = transform)
        preds_tensor, preds_label, preds_proba, labels, dataset, filenames = create_preds(dls, model, test_set = True)
        plt.figure()
        show_images_with_preds(preds_label, preds_proba, labels, filenames)
        plt.figure()
        show_histo_of_pred_tensor(preds_tensor)

    def report_results_accuracy(self, file_report_name, nb_of_iterations,source, lr, head_fine_tune, epoch_nb, batchsize, model_name, transform = True, balanced_fine_tune = False, preds_to_look_at = None):
        '''
        to save everything in a csv file
        '''
        acc_per_class = np.zeros((nb_of_iterations, 6))
        acc =  []

        if not os.path.exists(file_report_name):
            with open(file_report_name, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['model', 'test_set', 'fine_tune size', 'lr', 'head_fine_tune', 'balanced', 'preds_to_look_at', 'iteration', 'average_acc', 'basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'neutrophil', 'monocyte'])

        for i in [32,50,75,100]:
            for k in range(nb_of_iterations):
                model = torch.load('../models/Mixed_sources_trained/Barcelona_Saint_Antoine/resnet/' + model_name)
                model = workflow.fine_tune(model, source, batchsize, epoch_nb, lr, size = [i], transform = transform, head_fine_tune = head_fine_tune, balanced = balanced_fine_tune, preds_to_look_at = preds_to_look_at)
                res = workflow.show_confusion_matrix(model, source, transform = True, size = [5000])

                with open(file_report_name, 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([model_name, ', '.join(source), i,lr, head_fine_tune, balanced_fine_tune, preds_to_look_at, k+1, res[1]] + res[0].tolist())

    def report_results_clusters(self, model_name, file_report_name, source, transform, size, test_set = True, batchsize = 16, preds_to_look_at = None, balanced=None, show_graph = False, fine_tune = False, lr = None, fine_tune_size = None, epoch_nb = None):
        '''
        to evaluate quality of the clusters
        '''
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



    def knn_on_embeddings(self, model, source_train, source_test, batchsize = 32, size_training_set = [0,0,0,0], transform = False):
        embedding_train, labels_train, dataset_train, filenames_train = create_embeddings(self.entry_path, model, source_train, batchsize, test_set = False, transform = transform, size = size_training_set)
        embedding_test, labels_test, dataset_test, filenames_test = create_embeddings(self.entry_path, model, source_test, batchsize, test_set = True, transform = transform)
        knn = KNN_weighted(k = 5)
        knn.fit(np.array(embedding_train), np.array(labels_train))
        preds_test = knn.predict(embedding_test)
        plot_correlation(labels_test, preds_test, knn.classes_)
        print(sum(labels_test == preds_test)/len(labels_test))
        return labels_test, preds_test

    def label_with_siamese(self, model, unlabeled_source, unlabeled_size, labeled_source, labeled_size, list_labels_cat, transform = True, batchsize = 8, plot_matrix = True, normalize = True):
        real_label, pred_label = label_files(self.entry_path, unlabeled_source, unlabeled_size, labeled_source, labeled_size, model, list_labels_cat, transform = True, batchsize = 8)
        acc_per_class, acc = show_confusion_matrix(pred_label, real_label, list_labels_cat, plot_matrix,  normalize)
        return acc_per_class, acc
