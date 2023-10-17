from models.encoder_head_model import ModelFromArchitecture
from torch.utils.data.sampler import WeightedRandomSampler
from models.model_params import split_layers
from data.create_variables import create_dls
from fastai.optimizer import OptimWrapper
from torch import optim
import torch
import numpy as np
from fastai.vision.all import create_head, create_body, xresnet101, partial, Learner, LabelSmoothingCrossEntropy, accuracy, models
from efficientnet_pytorch import EfficientNet
from fastai.callback.core import Callback
from transformers import ViTFeatureExtractor, ViTForImageClassification

def create_model(siamese_head, architecture):
    if architecture == 'efficientnet':
        body = EfficientNet.from_pretrained('efficientnet-b0')
    if architecture == 'resnet':
        body = create_body(xresnet101(), cut=-4)
    if architecture == 'inception':
        body = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    if architecture == 'vgg':
        vgg_model = models.vgg16_bn(pretrained=True)
        body = create_body(vgg_model, cut=-2)
    if architecture == 'vit':
        body = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

    head = create_head(128, 12, ps=0.5)[2:]
    return ModelFromArchitecture(body, head, architecture)

class OverSamplingCallback(Callback):
    def __init__(self,learn:Learner):
        super().__init__(learn)
        self.labels = self.learn.data.train_dl.dataset.y.items
        _, counts = np.unique(self.labels,return_counts=True)
        self.weights = torch.DoubleTensor((1/counts)[self.labels])
        self.label_counts = np.bincount([self.learn.data.train_dl.dataset.y[i].data for i in range(len(self.learn.data.train_dl.dataset))])
        self.total_len_oversample = int(self.learn.data.c*np.max(self.label_counts))

    def on_train_begin(self, **kwargs):
        self.learn.data.train_dl.dl.batch_sampler = BatchSampler(WeightedRandomSampler(weights,self.total_len_oversample), self.learn.data.train_dl.batch_size,False)

def create_learner(model, dls, weights, wd = None):
    if wd is not None:
        learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(), splitter = split_layers, metrics = accuracy, wd = wd)
    else:
        learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(), splitter = split_layers, metrics = accuracy)
    if weights:
        # callback_fns = [partial(OverSamplingCallback)]
        callbacks = [OverSamplingCallback]
        learn.callbacks = callbacks
    return learn
