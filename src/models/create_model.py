from models.siamese.siamese_models import SiameseModel
from models.encoder_head_model import ModelFromResnet
from models.siamese.siamese_params import BCE_loss, siamese_splitter, my_accuracy
from models.model_params import split_layers
from data.create_variables import create_dls
from fastai.optimizer import OptimWrapper
from torch import optim
from fastai.vision.all import create_head, create_body, xresnet101, partial, Learner, LabelSmoothingCrossEntropy, accuracy

def create_model(siamese_head):
    body = create_body(xresnet101(), cut=-4)
    if siamese_head: 
        head = create_head(128, 1, ps=0.5)[2:]
        return SiameseModel(body, head)
    else: 
        head = create_head(128, 6, ps=0.5)[2:]
        return ModelFromResnet(body, head)

def create_learner(entry_path, model, source, siamese_head, batchsize = 32, size=0, transform = False):
    dls = create_dls(entry_path, source, siamese_head, batchsize = batchsize, size=size, transform = transform)
    if model.siamese_head:
        opt_func = partial(OptimWrapper, opt=optim.RMSprop)
        learn = Learner(dls, model, opt_func = opt_func, loss_func=BCE_loss, splitter = siamese_splitter, metrics=my_accuracy)                  
    else: 
        learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(), splitter = split_layers, metrics = accuracy)
    return learn
