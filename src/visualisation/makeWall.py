import torch
import torchvision
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
# from map_classes import dic_classes_st_antoinehttp://127.0.0.1:3005/?token=f51194683ad1099fa74a1f3fca576705d86d744833a700cb
import pickle
from torch import optim
from fastai.optimizer import OptimWrapper
import random
from sklearn.cluster import SpectralClustering
import PIL
import numpy as np
import fastai
from torch.nn import Conv2d
from torch.nn import MaxPool2d, AdaptiveMaxPool2d
# from torchmetrics.functional import pairwise_euclidean_distance
from torch.nn import Linear
from torch.nn import ReLU, LeakyReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import cv2
from fastai.vision.all import *
# from nbdev import show_doc

def create_df(dic_classes, source_df, list_labels_cat):


    dataframe = pd.read_csv('references/variables/dataframes/df_labeled_images.csv')


    dataframe_source0 = dataframe[dataframe['image_dataset'] == source_df ]
    dataframe_source = dataframe_source0.copy()
    for k in dic_classes:

        dataframe_source.loc[dataframe_source0['image_class']== k, 'image_class'] = dic_classes[k]

    print(dataframe_source['image_class'])
    dataframe_source = dataframe_source.loc[dataframe_source['image_class'].isin(list_labels_cat)]

    return dataframe_source


def open_image(fname, size=224):
    img = PIL.Image.open(fname).convert('RGB')
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    return t.permute(2,0,1).float()/255.0


def label_func(image_path, df):
    return list.index(list_labels_cat, df[df['image_path'] == image_path]['image_class'].iloc[0])

def import_variables(source1, source2):
    file = open('references/variables/dic_classes.obj', 'rb')
    dic_classes = pickle.load(file)

    list_labels_cat = ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'neutrophil'] if source1 =='matek' else ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'neutrophil', 'monocyte']
    list_labels = [0,1,2,3,4] if source1 == 'matek' else [0,1,2,3,4,5]


    dataframe_source = create_df(dic_classes,source2, list_labels_cat)

    files = list(dataframe_source['image_path'])
    array_files = np.array(files)


    class_list = list(dataframe_source['image_class'])
    array_class = np.array(class_list)

    file = open('references/variables/' + source1 + '_lbl2files.obj', 'rb')
    lbl2files = pickle.load(file)
    return dic_classes, list_labels_cat, list_labels, dataframe_source, files, array_files, class_list, array_class, lbl2files

def create_dataloader(source1):
    file = open('references/variables/' + source1 + '_tfm.obj', 'rb')
    tfm = pickle.load(file)
    file2 = open('references/variables/' + source1 + '_splits.obj', 'rb')
    splits = pickle.load(file2)
    trains  = array_files[splits[0]]
    valids = array_files[splits[1]]
    valids_class = array_class[splits[1]]
    # assert not [v for v in valids if v in array_files[splits[0]]]
    tls = TfmdLists(array_files, tfm, splits=splits)
    # dls = tls.dataloaders(after_item=[Resize(224), ToTensor],
    #                       after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])
    dls = tls.dataloaders(after_item=[Resize(224), ToTensor],
                        after_batch=[IntToFloatTensor], bs = 32)
    dls.cuda()
    return trains, valids, tls, dls, valids_class


class SiameseImage(fastuple):
    def show(self, ctx=None, **kwargs):
        if len(self) > 2:
            img1,img2,similarity = self
        else:
            img1,img2 = self
            similarity = 'Undetermined'
        if not isinstance(img1, Tensor):
            if img2.size != img1.size: img2 = img2.resize(img1.size)
            t1,t2 = tensor(img1),tensor(img2)
            t1,t2 = t1.permute(2,0,1),t2.permute(2,0,1)
        else: t1,t2 = img1,img2
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1,line,t2], dim=2), title=similarity, ctx=ctx, **kwargs)

class SiameseTransform(Transform):
    def __init__(self, files, splits):
        self.splbl2files = [{l: [f for f in files[splits[i]] if label_func(f, dataframe_source) == l] for l in list_labels}
                          for i in range(2)]
        self.valid = {f: self._draw(f,1) for f in files[splits[1]]}
        self.train = {f: self._draw(f,1) for f in files[splits[0]]}
    def encodes(self, f):
        f2,same = self.valid.get(f, self._draw(f,0))
        img1,img2 = PILImage.create(f),PILImage.create(f2)
        return SiameseImage(img1, img2, int(same))

    def _draw(self, f, split=0):
        same = random.random() < 0.5
        cls = label_func(f, dataframe_source)
        if not same: cls = random.choice(L(l for l in list_labels if l != cls))
        return random.choice(self.splbl2files[split][cls]),same


class SiameseModel(Module):
    def __init__(self, encoder, head):
        self.encoder,self.head = encoder,head

    # def forward(self, x1, x2):
    #     ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
    #     return self.head(ftrs)

    def similarity(self, x1, x2):
        x = torch.abs(x1 - x2)
        x = self.head(x)
        x = nn.Sigmoid()(x)
        return x

    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        return self.similarity(e1, e2)

def show_batch(x:SiameseImage, y, samples, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):
    if figsize is None: figsize = (ncols*6, max_n//ncols * 3)
    if ctxs is None: ctxs = get_grid(min(x[0].shape[0], max_n), nrows=None, ncols=ncols, figsize=figsize)
    for i,ctx in enumerate(ctxs): SiameseImage(x[0][i], x[1][i], ['Not similar','Similar'][x[2][i].item()]).show(ctx=ctx)

source1, source2 = 'saint_antoine', 'Saint_Antoine'
dic_classes, list_labels_cat, list_labels, dataframe_source, files, array_files, class_list, array_class, lbl2files = import_variables(source1, source2)
trains, valids, tls, dls, valids_class = create_dataloader(source1)
model2 = torch.load('models/Saint_Antoine_trained/siamese/siamese8_stage1', map_location = 'cpu')
def create_embeddings(Nimages, batch):

    list_image = []
    print(len(valids))
    for i in range(Nimages*(batch-1), Nimages*batch):
        print(i, end = '\r')
        t1 = open_image(valids[i])
        list_image.append(model2.encoder(t1.unsqueeze(0)))
    return list_image

def create_resdistance(Nimages, batch):
    res = np.zeros((Nimages, Nimages))
    list_image = create_embeddings(Nimages, batch)
    for i in range(Nimages):
        for j in range(i, Nimages):
            print(i, end = '\r')
            t1 = list_image[i]
            t2 = list_image[j]
            out = torch.abs(t1 - t2)
            out = model2.head(out)
            out = nn.Sigmoid()(out)

            res[i][j] = out.item()
            res[j][i] = res[i][j]
    return res
# res = create_resdistance(500, 2)
# file = open('references/variables/saint_antoine_embeddings_si8.obj', 'wb')
# pickle.dump(list_image, file)
# print(list_image[0].shape)
# file = open('references/variables/saint_antoine_embeddings_si8.obj', 'rb')
# list_image = pickle.load(file)

# file = open('references/variables/matek_resdistance_si8.obj', 'wb')
# pickle.dump(res, file)
file = open('references/variables/' + source1 + '_resdistance_si8.obj', 'rb')
res = pickle.load(file)
print(res)


def makeWall(im, labels, order, side=7):

    Nimage = im.shape[0]
    yside = side
    xside = 1 + Nimage//side

    res =  np.zeros(250*250*xside*yside*3).reshape(250*xside, 250*yside, 3)

    for i in range(Nimage):
        img = np.array(im[[order[i]]][0,:,:,:])
#         print(np.shape(img))
        # cv2.rectangle(img,(250,0),(0,250),(0,255,0),20)
        font = cv2.FONT_HERSHEY_PLAIN

#         cv2.putText(img,labels[[order[i]]][0],(10,250), font, 1,(0,255,0),1)
        cv2.putText(img,labels[[order[i]]][0],(10,250), font, 2,(0,0,0),1)
        norm = 1
        if source1 == 'matek':
            norm = 255
        res[250*(i//side):250*(i//side)+250, 250*(i%side):250*(i%side)+250] = img/norm
    return(res)

def make_cluster(res, Nimages, batch):
    Ninit = 1000
    Ncluster = 8
    sc = SpectralClustering(Ncluster, affinity='precomputed', n_init=Ninit, assign_labels='kmeans')
    a = sc.fit_predict(res)
    indiceCluster = [np.arange(Nimages)[a==i] for i in range(Ncluster)]
#     os.makedirs("matek_walls")
    allIm = np.array([plt.imread(x)[:,:,:3] for x in valids[(batch-1)*Nimages : batch*Nimages]])
    for i in range(Ncluster):
        print("Walls: " + str(i+1) + "/" + str(Ncluster), end = '\r')
        distanceIntraCluster =  np.mean(res[np.ix_(indiceCluster[i], indiceCluster[i])], axis=0)
        order = np.flip(np.argsort(distanceIntraCluster))

        newWall = makeWall(allIm[indiceCluster[i]], valids_class[indiceCluster[i]+Nimages*(batch-1)], order)
        plt.imsave(arr= newWall, fname = 'reports/' + source1 + "_walls/saint_antoine_training/batch_"+ str(batch)+ "/cluster" + str(i) + ".png")

make_cluster(res, 500, 1)
