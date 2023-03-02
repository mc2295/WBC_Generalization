import torch
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
import matplotlib.pyplot as plt
from fastai.optimizer import OptimWrapper
from torch import optim
from pylab import rcParams
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from fastai.vision.all import *
from data.siamese_image import SiameseTransform, show_batch
from models.siamese.siamese_models import SiameseModel
from models.encoder_head_model import Model, ModelFromResnet
from data.create_variables import create_variables, create_dataloader, create_dataloader_siamese
from models.siamese.siamese_params import BCE_loss, siamese_splitter, my_accuracy
from visualisation.clusters import scatter_plot_2D
from visualisation.KNN import plot_correlation
# from nbdev import show_doc

class Workflow():
    def __init__(self, entry_path):
        self.entry_path = entry_path
    def create_dls(self, source, batchsize, siamese_head):
        if siamese_head: 
            return create_dataloader_siamese(self.entry_path, source, batchsize, SiameseTransform)
        else: 
            return create_dataloader(self.entry_path, source, batchsize)

    def create_model(self, siamese_head):
        body = create_body(xresnet101(), cut=-4)
        if siamese_head: 
            head = create_head(128, 1, ps=0.5)[2:]
            return SiameseModel(body, head)
        else: 
            head = create_head(128, 6, ps=0.5)[2:]
            return ModelFromResnet(body, head)
    
    def import_model(self, model_path):
        model = torch.load(model_path)
        return model

    def create_learner(self, model, dls):
        if model.siamese_head:
            opt_func = partial(OptimWrapper, opt=optim.RMSprop)
            learn = Learner(dls, model, opt_func = opt_func, loss_func=BCE_loss, splitter = siamese_splitter, metrics=my_accuracy)                  
        else: 
            learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(), metrics = accuracy)
        return learn
    
    def train_model(self, source_train, batchsize, epoch_nb, siamese_head, lr = None):
        model = self.create_model(siamese_head)
        dls = self.create_dls(source_train, batchsize, siamese_head)
        learn = self.create_learner(model, dls)
        if lr is not None: 
            learn.fit_one_cycle(epoch_nb, lr)
        else: 
            learn.lr_find()
            print(learn.lr_find())
        return learn.model

    def evaluate_model(self, model, source_test, batchsize):
        dls_test = self.create_dls(source_test, batchsize, model.siamese_head)
        learn_test = self.create_learner(model, dls_test)
        preds, target = learn_test.get_preds(dl = dls_test.valid)
        if model.siamese_head == False:
            interp = ClassificationInterpretation.from_learner(learn_test)
            interp.plot_confusion_matrix(normalize=True)
        return accuracy(preds, target)

    def add_classifier_head(self, model):
        return Model(model.encoder, create_head(128, 6, ps=0.5)[2:], siamese_head = False)

    def fine_tune(self, model, source_tuning, batchsize, epoch_nb, lr = None):
        dls_tune = self.create_dls(source_tuning, batchsize, model.siamese_head)
        learn = self.create_learner(model, dls_tune)
        learn.freeze_to(1)
        if lr: 
            learn.fit_one_cycle(epoch_nb, lr)
        else: 
            learn.lr_find()
            print(learn.lr_find())
        return learn.model
    
    def create_embeddings(self, model, source, batchsize):
        dls = self.create_dls(source, batchsize, siamese_head = False)
        learn = Learner(dls, model.encoder, loss_func = LabelSmoothingCrossEntropy())
        embedding, labels = learn.get_preds(dl=dls.train)
        return embedding, labels
    
    def visualise_2D(self, model, source, batchsize, method):
        embedding, labels = self.create_embeddings(model, source, batchsize)
        X_proj = scatter_plot_2D(embedding, labels, method)
        return X_proj

    def knn_on_embeddings(self, model, source_train, source_test, batchsize):
        embedding_train, labels_train = self.create_embeddings( model, source_train, batchsize)
        embedding_test, labels_test = self.create_embeddings( model, source_test, batchsize)
        knn = KNeighborsClassifier(n_neighbors = 4, metric='manhattan')
        knn.fit(embedding_train, labels_train)
        preds_test = knn.predict(embedding_test)
        print(knn.score(embedding_test, labels_test))
        list_labels_cat = ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'neutrophil', 'monocyte']
        plot_correlation(labels_test, preds_test, list_labels_cat)
        return preds_test
