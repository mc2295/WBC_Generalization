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
from models.siamese.siamese_models import SiameseModel_gregoire, SiameseModel_encoder_head
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
    print(fname)
    img = PIL.Image.open('../../Documents/These/' + fname).convert('RGB')
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    return t.permute(2,0,1).float()/255.0


def label_func(image_path, df):
    return list.index(list_labels_cat, df[df['image_path'] == image_path]['image_class'].iloc[0])

def import_variables(source1, source2):
    file = open('references/variables/dic_classes.obj', 'rb')
    dic_classes = pickle.load(file)

    list_labels_cat = (['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'neutrophil']
                    if source1 =='matek'
                    else ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'neutrophil', 'monocyte'])
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
    # assert not [v for v in valids if v in array_files[splits[0]]]
    tls = TfmdLists(array_files, tfm, splits=splits)
    # dls = tls.dataloaders(after_item=[Resize(224), ToTensor],
    #                       after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])
    dls = tls.dataloaders(after_item=[Resize(224), ToTensor],
                        after_batch=[IntToFloatTensor], bs = 32)
    dls.cuda()
    return trains, valids, tls, dls


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
        img1,img2 = PILImage.create('../../Documents/These/' + f),PILImage.create('../../Documents/These/' + f2)
        return SiameseImage(img1, img2, int(same))

    def _draw(self, f, split=0):
        same = random.random() < 0.5
        cls = label_func(f, dataframe_source)
        if not same: cls = random.choice(L(l for l in list_labels if l != cls))
        return random.choice(self.splbl2files[split][cls]),same

def show_batch(x:SiameseImage, y, samples, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):
    if figsize is None: figsize = (ncols*6, max_n//ncols * 3)
    if ctxs is None: ctxs = get_grid(min(x[0].shape[0], max_n), nrows=None, ncols=ncols, figsize=figsize)
    for i,ctx in enumerate(ctxs): SiameseImage(x[0][i], x[1][i], ['Not similar','Similar'][x[2][i].item()]).show(ctx=ctx)

source1, source2 = 'saint_antoine', 'Saint_Antoine'
dic_classes, list_labels_cat, list_labels, dataframe_source, files, array_files, class_list, array_class, lbl2files = import_variables(source1, source2)
trains, valids, tls, dls = create_dataloader(source1)

# b = dls.one_batch()
# show_batch(b,None,None)
# model_meta[xresnet34]
def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]

def CrossEnt_loss(out, targ):
    return CrossEntropyLossFlat()(out, targ.long())

def MCE_loss(out, target):
    res = (out - target).pow(2).mean()
    return res

def BCE_loss(out, target):

    return nn.BCELoss()(torch.squeeze(out, 1), target.float())


def contrastive_loss(y_pred, y_true):

    margin =1
    label = (y_pred > 0.5).squeeze(1).float()
    label = torch.tensor(label, requires_grad = True)
    square_pred = torch.square(label)
    a = torch.tensor(margin, dtype=torch.int8) - (label)
    b = torch.tensor(0, dtype = torch.int8).type_as(label)

    # margin_square = torch.square(torch.maximum(torch.tensor(margin, dtype=torch.int8) - (y_pred), torch.tensor(0, dtype=torch.int8)))
    margin_square = torch.square(torch.maximum(a, b))
    return torch.mean(
        (1 - y_true) * square_pred + (y_true) * margin_square
    )

def my_accuracy(input, target):
    label = input > 0.5
    return (label.squeeze(1) == target).float().mean()

def my_accuracy_2(output, target):
    euclidean_distance = F.pairwise_distance(output[0], output[1], keepdim = True)
    label = euclidean_distance > 0.5
    return (label.squeeze(1) == target).float().mean()


class ContrastiveLoss2(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss2, self).__init__()
        self.margin = margin

    def forward(self, output, label):

        # Calculate the euclidean distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output[0], output[1], keepdim = True)
        print('label', label)

        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        print('distance', euclidean_distance)
        return loss_contrastive

opt_func = partial(OptimWrapper, opt=optim.RMSprop)

# contrastive_loss2 = ContrastiveLoss2().cuda()
# encoder = nn.Sequential(
#     create_body(xresnet34(), cut=-2),
#     AdaptiveConcatPool2d(),
#     nn.Flatten()
# )
# head = nn.Linear(1024, 1)
encoder = create_body(xresnet101(), cut=-4)
head = create_head(512*4, 1, ps=0.5)


model = SiameseModel_encoder_head(encoder, head)

learn = Learner(dls, model, opt_func = opt_func, loss_func=BCE_loss, splitter=siamese_splitter, metrics=my_accuracy)
# learn = Learner(dls, model, opt_func = opt_func, loss_func=my_loss, metrics=my_accuracy)

learn.fit_one_cycle(1, slice(1e-6,1e-4))

# learn.freeze()
# learn.lr_find()
# learn.fit_one_cycle(4, 3e-3)
# learn.unfreeze()
torch.save(learn.model, 'models/'+ source2 + '_trained/siamese/siamese_test')

# learn.fine_tune(2)


# import gc
# gc.collect()
# torch.cuda.empty_cache()
# learn1.predict()


def treat(img, size=224):
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    return t.permute(2,0,1).float()/255.0

def check_accuracy():
    correct = 0
    for k in range(200):

        im1 = treat(tls.valid[k][0])
        im2 = treat(tls.valid[k][1])
        result = learn.model(im1.unsqueeze(0).cuda(), im2.unsqueeze(0).cuda())
        print(k,  end = '\r')
        if int(np.round(result.item()))== tls[k][2]:
            correct+=1
    return correct

# print(correct)
print(check_accuracy())
print(len(valids))
@typedispatch
def show_results(x:SiameseImage, y, samples, outs, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):
    if figsize is None: figsize = (ncols*6, max_n//ncols * 3)
    if ctxs is None: ctxs = get_grid(min(x[0].shape[0], max_n), nrows=None, ncols=ncols, figsize=figsize)
    for i,ctx in enumerate(ctxs):
        # print(x[1][i]== y[1][i], y[1][i], y[2][i])
        title = f'Actual: {["Not similar","Similar"][int(np.round(x[2][i].item()))]} \n Prediction: {["Not similar","Similar"][int(y[2][i].item())]}'
        # title = f'Actual: {["Not similar","Similar"][x[i].item()]} \n Prediction: {["Not similar","Similar"][y[i].item()]}'
        SiameseImage(x[0][i], x[1][i], title).show(ctx=ctx)


learn.show_results()
