import numpy as np
from PIL import Image
import torch as torch
import torch.nn as nn
from fastai.vision.all import Learner, ImageDataLoaders, Resize
import pickle
import pandas as pd

def open_image(fname, size=224):
    # print(fname)
    img = Image.open('../../Documents/These/' + fname).convert('RGB')
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    return t.permute(2,0,1).float()/255.0


def create_embeddings(model, array_files, array_class, splits):
    df = pd.DataFrame(list(zip(array_files, array_class, [i in splits[1] for i in range(len(array_files))])), columns = ['name', 'label', 'is_valid'])
    dls = ImageDataLoaders.from_df(df, item_tfms=Resize(224),  path = '../../Documents/These/', valid_col='is_valid')
    if torch.cuda.is_available():
        dls.cuda()

    learn = Learner(dls, model.encoder)

    test_dl = dls.valid
    embedding_valids, targ_valids = learn.get_preds(dl=test_dl)

    train_dl = dls.train
    embedding_trains, targ_trains = learn.get_preds(dl=train_dl)


    return embedding_valids, targ_valids, embedding_trains, targ_trains

def import_embeddings(source):
    file1 = open('references/variables/' + source + '_embedding_trains.obj', 'rb')
    file2 = open('references/variables/' + source + '_embedding_valids.obj', 'rb')
    file3 = open('references/variables/' + source + '_targ_trains.obj', 'rb')
    file4 = open('references/variables/' + source + '_targ_valids.obj', 'rb')
    embedding_trains, embedding_valids, targ_trains, targ_valids = pickle.load(file1), pickle.load(file2), pickle.load(file3), pickle.load(file4)
    return embedding_trains, embedding_valids, targ_trains, targ_valids

def create_resdistance(Nimages, batch, valids, model):
    res = np.zeros((Nimages, Nimages))
    list_image = create_embeddings(Nimages, batch, valids, model)
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
