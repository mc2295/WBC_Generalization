from models.siamese.siamese_models import SiameseModel
from models.encoder_head_model import ModelFromResnet
from torch.utils.data.sampler import WeightedRandomSampler
from models.siamese.siamese_params import BCE_loss, siamese_splitter, my_accuracy
from models.model_params import split_layers
from data.create_variables import create_dls
from fastai.optimizer import OptimWrapper
from torch import optim
import torch
import numpy as np
from fastai.vision.all import create_head, create_body, xresnet101, partial, Learner, LabelSmoothingCrossEntropy, accuracy
from fastai.callback.core import Callback

def create_model(siamese_head):
    body = create_body(xresnet101(), cut=-4)
    if siamese_head:
        head = create_head(128, 1, ps=0.5)[2:]
        return SiameseModel(body, head)
    else:
        head = create_head(128, 6, ps=0.5)[2:]
        return ModelFromResnet(body, head)


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

def create_learner(model, dls, weights):
    if model.siamese_head:
        opt_func = partial(OptimWrapper, opt=optim.RMSprop)
        learn = Learner(dls, model, opt_func = opt_func, loss_func=BCE_loss, splitter = siamese_splitter, metrics=my_accuracy)
    else:
        learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(), splitter = split_layers, metrics = accuracy)
        if weights:
            # callback_fns = [partial(OverSamplingCallback)]
            callbacks = [OverSamplingCallback]
            learn.callbacks = callbacks
    return learn
