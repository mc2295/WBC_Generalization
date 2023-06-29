import numpy as np
from PIL import Image
import torch as torch
import torch.nn as nn
from fastai.vision.all import Learner, ImageDataLoaders, Resize, LabelSmoothingCrossEntropy, ClassificationInterpretation, to_np, flatten_check, noop, accuracy
import pickle
import pandas as pd
from models.siamese.siamese_params import BCE_loss, siamese_splitter, my_accuracy
from models.model_params import split_layers
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
# from umap import UMAP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from data.create_variables import create_dls, list_labels_cat
from visualisation.create_plots import project_2D, plot_confusion_matrix

'''
In this module:
- open_image : ok
- flatten_batch : creates a list of all flattened images of a batch
- create_matrix_of_flattened_images : same but for multiple sources
- create_embeddings : creates list of embeddings through a model, with associated label and dataset
- create_filenames_from_dls : if I want to access to names of images of training set from a dataloader
'''


def open_image(fname, size=224):
    # print(fname)
    img = Image.open('../../Documents/These/' + fname).convert('RGB')
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    return t.permute(2,0,1).float()/255.0

def flatten_batch(b):
    n = len(b)
    X = torch.empty(n, 224*224)
    for j in range(n):
        X[j] = torch.sum(b[j], dim = 0).view(224*224)
    return X

def create_matrix_of_flattened_images(entry_path, batchsize, source, transform):
    n = batchsize
    X = torch.empty(n*len(source),224*224)
    dataset = []
    for i,k in enumerate(source):
        dls = create_dls(entry_path, [k], siamese_head = False, batchsize= n, transform = transform)
        b = dls.one_batch()[0]
        b = flatten_batch(b)
        X[i*n: (i+1)*n] = b
        dataset += [k for i in range(n)]
    return X, dataset

def run_through_model(dls, model, siamese_head = False, test_set = False):
    if siamese_head:
        learn = Learner(dls, model, splitter = siamese_splitter, loss_func = noop)
        out,targs = learn.get_preds(dl = dls.valid if test_set else dls.train)
        return out, targs
    else:
        learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(), splitter = split_layers, metrics = accuracy)
        out,targs = learn.get_preds(dl = dls.valid if test_set else dls.train)
        ds = learn.dls.valid_ds if test_set else learn.dls.train_ds
        labels = [str(ds.label[j]) for j in range(len(out))]
        filenames = [str(ds.name[j]) for j in range(len(out))]
        dataset = [ds.dataset[j] for j in range(len(out))]

        return out, labels, dataset, filenames

def create_embeddings(model, dls, test_set = False):
    model = model.encoder
    X, labels, dataset, filenames = run_through_model(dls, model, siamese_head = False, test_set = test_set)
    return X, labels, dataset, filenames

def create_preds(dls, model, test_set = False, preds_to_look_at = None):
    y = run_through_model(dls, model, model.siamese_head, test_set = test_set)
    if model.siamese_head:
        preds, targs = y
        decoded = preds > 0.5
        preds = decoded.int()
        return preds, targs
    else:
        preds_tensor, labels, dataset, filenames = y
        preds_num = preds_tensor.argmax(dim=1)
        preds_label = [list_labels_cat[i] for i in preds_num]
        preds_proba = torch.max(preds_tensor, dim = 1)
        if preds_to_look_at is not None:
            zipped = zip(preds_tensor, preds_label, preds_proba, labels, dataset, filenames)
            if preds_to_look_at == 'worst_preds':
                reverse = False
            elif preds_to_look_at == 'best_preds':
                reverse = True
            zipped = sorted(zipped, key = lambda t: t[2], reverse = reverse)
            preds_tensor, preds_label, preds_proba, labels, dataset, filenames = [zipped[i][0] for i in range(500)], [zipped[i][1] for i in range(500)],[zipped[i][2] for i in range(500)],[zipped[i][3] for i in range(500)],[zipped[i][4] for i in range(500)],[zipped[i][5] for i in range(500)]

        return preds_tensor, preds_label, preds_proba, labels, dataset, filenames



def dist_intra_inter_cluster(data, labels, embedding_name):
    intra_cluster_distances = []
    for cluster_label in np.unique(labels):
        cluster_points = data.loc[data.labels == cluster_label][embedding_name]
        cluster_distance = pairwise_distances(cluster_points.tolist())
        average_distance = np.mean(cluster_distance)
        intra_cluster_distances.append(average_distance)

    # Calculate inter-cluster distance
    inter_cluster_distances = []
    for cluster_label1 in np.unique(labels):
        for cluster_label2 in np.unique(labels):
            if cluster_label1 != cluster_label2:
                cluster_points1 = data.loc[data.labels == cluster_label1][embedding_name].tolist()
                cluster_points2 = data.loc[data.labels == cluster_label2][embedding_name].tolist()

                distance = pairwise_distances(cluster_points1, cluster_points2)
                average_distance = np.mean(distance)
                inter_cluster_distances.append(average_distance)

    # Calculate the average intra-cluster distance and inter-cluster distance
    average_intra_cluster_distance = np.mean(intra_cluster_distances)
    average_inter_cluster_distance = np.mean(inter_cluster_distances)
    return average_intra_cluster_distance, average_inter_cluster_distance

def evaluate_cluster(embedding, labels, size):
    nb = 0
    for i in size:
        nb+=i

    X = embedding.detach().numpy()
    X_proj = project_2D(X,labels,'t-SNE')
        # Assuming you have your data points stored in 'data' and cluster labels in 'labels'
    silhouette, silhouette_2D = metrics.silhouette_score(X, labels, metric='euclidean'), metrics.silhouette_score(X_proj, labels, metric='euclidean')
    calinski, calinski_2D = metrics.calinski_harabasz_score(X, labels), metrics.calinski_harabasz_score(X_proj, labels)
    bouldin, bouldin_2D = metrics.davies_bouldin_score(X, labels), metrics.davies_bouldin_score(X_proj, labels)

    data = pd.DataFrame(
        dict(embedding_2D = [X_proj[i] for i in range(nb)],
        embedding = [X[i] for i in range(nb)],
        labels = labels))

    average_intra_cluster_distance, average_inter_cluster_distance = dist_intra_inter_cluster(data, labels, 'embedding')
    average_intra_cluster_distance_2D, average_inter_cluster_distance_2D = dist_intra_inter_cluster(data, labels, 'embedding_2D')

    return silhouette, bouldin, calinski, average_intra_cluster_distance, average_inter_cluster_distance, silhouette_2D, bouldin_2D, calinski_2D, average_intra_cluster_distance_2D, average_inter_cluster_distance_2D
    # Calculate intra-cluster distance

def show_confusion_matrix(pred_label, real_label, list_labels, plot_matrix, normalize):
    cm = confusion_matrix(real_label, pred_label[:len(real_label)],  labels = list_labels, normalize = 'true')
    if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_without_nan = np.nan_to_num(cm)
    acc = cm_without_nan.diagonal().sum()/cm_without_nan.sum()
    acc_per_class = cm.diagonal()/cm_without_nan.sum(axis=1)
    if plot_matrix:
        plot_confusion_matrix(cm, list_labels, normalize)
    return acc_per_class, acc
