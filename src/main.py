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
from data.create_variables import create_dataloader, create_variables
from data.siamese_image import open_image, label_func, SiameseImage, SiameseTransform
from models.siamese.siamese_params import BCE_loss, siamese_splitter, my_accuracy, contrastive_loss
from visualisation.check_accuracy import check_accuracy
from visualisation.make_embeddings import create_resdistance, create_embeddings
from visualisation.clusters import scatter_plot_clusters, make_cluster
# from nbdev import show_doc


source1, source2 = 'saint_antoine', 'Saint_Antoine'
dic_classes, list_labels_cat, list_labels, dataframe_source, files, array_files, class_list, array_class = create_variables('references', source1, source2)

trains, valids, valids_class, tls, dls = create_dataloader('references', source1, array_files, array_class,SiameseTransform, dataframe_source, list_labels_cat, list_labels)

opt_func = partial(OptimWrapper, opt=optim.RMSprop)

# encoder = create_body(xresnet101(), cut=-4)
# head = create_head(512*4, 1, ps=0.5)

# model = SiameseModel(encoder, head)

# learn = Learner(dls, model, opt_func = opt_func, loss_func=BCE_loss, splitter=siamese_splitter, metrics=my_accuracy)

# learn.fit_one_cycle(1, slice(1e-6,1e-4))

# torch.save(learn.model, 'models/'+ source2 + '_trained/siamese/siamese_test')

# print(check_accuracy(tls, learn, 200))
training_source1, training_source2 = 'barcelona', 'Barcelona'
siamese_number = str(1)
model = torch.load('models/'+ training_source2 + '_trained/siamese' + siamese_number + '_stage1', map_location = 'cpu')

# print(valids[0])
# print(check_accuracy(tls, model, 200))
# list = create_embeddings(99, 1, valids, model)
# res = create_resdistance(150, 5, valids, model)

file = open('references/variables/resdistance_' + training_source1 + '_si'+ siamese_number + '_on_' + source1 + '_batch_1.obj', 'rb')
res = pickle.load(file)

make_cluster(res, 500, 1, valids, valids_class, source1,training_source1)

scatter_plot_clusters(res, valids_class, 500, 1)
