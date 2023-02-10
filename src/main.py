import torch
import torchvision
from PIL import Image
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
# from map_classes import dic_classes_st_antoinehttp://127.0.0.1:3005/?token=f51194683ad1099fa74a1f3fca576705d86d744833a700cb
import pickle
from torch import optim
from fastai.optimizer import OptimWrapper
import random
import PIL
import numpy as np
from torch.nn import Conv2d
from torch.nn import Module
from torch.nn import MaxPool2d, AdaptiveMaxPool2d
# from torchmetrics.functional import pairwise_euclidean_distance
from torch.nn import Linear
from torch.nn import ReLU, LeakyReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from fastai.vision.all import *
# from dataloader import SiameseImage, SiameseTransform
from models.siamese.siamese_models import SiameseModel_gregoire, SiameseModel
from data.create_variables import create_siamese_dataloader, create_variables
from data.siamese_image import open_image, label_func, SiameseImage, SiameseTransform
from models.siamese.siamese_params import BCE_loss, siamese_splitter, my_accuracy, contrastive_loss
from visualisation.check_accuracy import check_accuracy
from visualisation.make_embeddings import create_resdistance, create_embeddings, import_embeddings, project_2D
from visualisation.clusters import scatter_plot_clusters, make_walls
from visualisation.KNN import KNN_score, plot_correlation, plot_KNN_space
# from nbdev import show_doc


source = ['barcelona']
array_files, array_class, splits, dic_labels, list_labels_cat = create_variables('references', source)

### train model
# trains, valids, valids_class, tls, dls = create_siamese_dataloader(array_files, array_class, splits, dic_labels, list_labels_cat, SiameseTransform, 8)

# opt_func = partial(OptimWrapper, opt=optim.RMSprop)

# encoder = create_body(xresnet101(), cut=-4)
# head = create_head(128, 1, ps=0.5)[2:]

# model = SiameseModel(encoder, head)

# learn = Learner(dls, model, opt_func = opt_func, loss_func=BCE_loss, splitter=siamese_splitter, metrics=my_accuracy)

# learn.fit_one_cycle(1, slice(1e-6,1e-4))

# torch.save(learn.model, 'models/'+ source2 + '_trained/siamese/siamese_test')


### visualise

# training_source1, training_source2 = 'saint_antoine'
# siamese_number = str(8)

# model = torch.load('models/'+ training_source2 + '_trained/siamese' + siamese_number + '_stage1', map_location = 'cpu')

# print(check_accuracy(dls, model))

embedding_trains, embedding_valids, targ_trains, targ_valids = import_embeddings(source[0])
X = torch.cat((embedding_trains, embedding_valids))
y = torch.cat((targ_trains, targ_valids))

# preds_valids, KNN_accuracy = KNN_score(embedding_trains, embedding_valids, targ_trains, targ_valids)


# plot_correlation(targ_valids, preds_valids, list_labels_cat)

X_2D = project_2D(X,y,'t-SNE')
# scatter_plot_clusters(X_2D, y, 't-SNE')

plot_KNN_space(targ_trains, targ_valids, X_2D, y)

# res = create_resdistance(150, 5, valids, model)

# make_walls(res, 500, 1, valids, valids_class, source1,training_source1)
