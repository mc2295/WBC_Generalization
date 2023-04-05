import numpy as np
from PIL import Image
import torch as torch
import torch.nn as nn
from fastai.vision.all import Learner, ImageDataLoaders, Resize
import pickle
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
# from umap import UMAP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from data.create_variables import create_dls

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

def create_embeddings(entry_path, model, source, batchsize = 32, test_set = False, transform = False, size = 0):
    X = []
    labels = []
    dataset = []
    for k in source:
        if k == 'matek' and size == 0:
            size = 15000
        dls = create_dls(entry_path, [k],siamese_head= False, batchsize = batchsize, size = size, transform = transform)
#             learn = Learner(dls, model.encoder, loss_func = LabelSmoothingCrossEntropy())
        learn = Learner(dls, model.encoder)
#             learner = create_learner(self.entry_path, model.encoder, source_train, False, batchsize = batchsize, size = size, transform = transform)
        dl_set = dls.valid if test_set else dls.train
        embedding_trains, label = learn.get_preds(dl=dl_set) 
        n = embedding_trains.shape[0]
        X+= [embedding_trains[j].detach().numpy() for j in range(n)]
        labels+= [label[j].detach().numpy() for j in range(n)]
        dataset += [k for i in range(n)] 
    return X, labels, dataset

def create_filenames_from_dls(entry_path, model, source, batchsize=32):
    filenames = []
    for k in source:
        if k == 'matek':
            dls = create_dls(entry_path,[k], siamese_head = False, batchsize = batchsize, size = 15000)
        else: 
            dls = create_dls(entry_path, [k], siamese_head = False, batchsize = batchsize)
        learn = Learner(dls, model.encoder)
        for i in range(len(learn.dls.train_ds)):
            filenames.append(str(learn.dls.train_ds.items.name[i]))
    return filenames