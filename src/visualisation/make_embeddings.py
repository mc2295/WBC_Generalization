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
from sklearn import (manifold, decomposition)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def open_image(entry_path, fname, size=224):
    # print(fname)
    img = Image.open(entry_path+ fname).convert('RGB')
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    return t.permute(2,0,1).float()/255.0

def import_embeddings(source):
    file1 = open('references/variables/' + source + '_embedding_trains.obj', 'rb')
    file2 = open('references/variables/' + source + '_embedding_valids.obj', 'rb')
    file3 = open('references/variables/' + source + '_targ_trains.obj', 'rb')
    file4 = open('references/variables/' + source + '_targ_valids.obj', 'rb')
    embedding_trains, embedding_valids, targ_trains, targ_valids = pickle.load(file1), pickle.load(file2), pickle.load(file3), pickle.load(file4)
    return embedding_trains, embedding_valids, targ_trains, targ_valids

def create_resdistance(entry_path, Nimages, batch, valids, model):
    list_image = []
    print(len(valids))
    for i in range(Nimages*(batch-1), Nimages*batch):
        print(i, end = '\r')
        t1 = open_image(entry_path, valids[i])
        list_image.append(model.encoder(t1.unsqueeze(0)))

    res = np.zeros((Nimages, Nimages))
    for i in range(Nimages):
        for j in range(i, Nimages):
            print(i, end = '\r')
            t1 = list_image[i]
            t2 = list_image[j]
            out = torch.abs(t1 - t2)
            out = model.head(out)
            out = nn.Sigmoid()(out)

            res[i][j] = out.item()
            res[j][i] = res[i][j]
    return res
