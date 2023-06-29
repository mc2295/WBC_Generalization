import torch
import matplotlib.pyplot as plt
import numpy as np
from fastai.vision.all import create_head, ClassificationInterpretation, accuracy
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from data.siamese_image import SiameseTransform, show_batch
from data.create_variables import create_variables, create_dls
from models.create_model import create_model, create_learner, create_dls
from models.encoder_head_model import Model
from visualisation.clusters import scatter_plot_embeddings, create_df_of_2D_embeddings_info, visualise_images_in_region
from visualisation.KNN import plot_correlation
from visualisation.make_embeddings import flatten_batch, create_matrix_of_flattened_images, create_embeddings, create_filenames_from_dls

# from nbdev import show_doc
print(torch.cuda.is_available())
class Workflow():
    def __init__(self, entry_path):
        self.entry_path = entry_path

    def find_learning_rate(self, model, source, batchsize = 32, size = 0, transform = False):
        learn = create_learner(self.entry_path, model, source, model.siamese_head, batchsize, size = size, transform = transform)
        learn.lr_find()
        print(learn.lr_find())
        return learn.lr_find()

    def train_model(self, source_train, epoch_nb, siamese_head,  lr, batchsize = 32, size = 0, transform = False):
        model = create_model(siamese_head)
        learn = create_learner(self.entry_path, model, source_train, siamese_head, batchsize, size = size, transform = transform)
        learn.fit_one_cycle(epoch_nb, lr)
        return learn.model

    def fine_tune_with_classifier_head(self, model, source_tuning, batchsize, epoch_nb, lr, size = 0, transform = False):
        model = Model(model.encoder, create_head(128, 6, ps=0.5)[2:], siamese_head = False)
        learn = create_learner(self.entry_path, model, source_tuning, siamese_head = False, batchsize = batchsize, size = size, transform = transform)
        learn.freeze()
        learn.fit_one_cycle(epoch_nb, lr)
        return learn.model

    def visualise_datasets_intensity_2D(self, image_nb_per_source, source, transform = False):
        X, dataset = create_matrix_of_flattened_images(self.entry_path, image_nb_per_source, source, transform = transform)
        labels = []
        data = scatter_plot_embeddings(X, labels, 't-SNE', dataset, display_classes = False)
        return data

    def visualise_embeddings_2D(self, model, source, method, display_classes = True, batchsize = 32,test_set = False, transform = False):
        embedding, labels, dataset = create_embeddings(self.entry_path, model, source, batchsize, test_set, transform = transform)
        df_embeddings = scatter_plot_embeddings(embedding, labels, method, dataset, display_classes)
        return df_embeddings

    def display_images_from_scatter_plot_region(self, source, model, df_embeddings, cell_class, x_region, y_region):
        filenames = create_filenames_from_dls(self.entry_path, model, source)
        df_embeddings['name'] = filenames
        group = visualise_images_in_region(self.entry_path, df_embeddings, cell_class, x_region, y_region)
        return group

    def show_confusion_matrix(self, model, source_test, batchsize = 32, size = 0, transform = False):
        learn = create_learner(self.entry_path, model, source_test, model.siamese_head, batchsize, size = size, transform = transform)
        preds, target, dataset = create_embeddings(self.entry_path, model, source_test, batchsize, test_set = True, transform = transform)
        if model.siamese_head == False:
            interp = ClassificationInterpretation.from_learner(learn)
            interp.plot_confusion_matrix(normalize=True)
        return accuracy(torch.tensor(np.array(preds)), torch.tensor(np.array(target)))


    def knn_on_embeddings(self, model, source_train, source_test, batchsize = 32, size_training_set = 0, transform = False):
        embedding_train, labels_train , dataset_train = create_embeddings(self.entry_path, model, source_train, batchsize, test_set = False, transform = transform)
        print(len(embedding_train))
        embedding_test, labels_test, dataset_test = create_embeddings(self.entry_path, model, source_test, batchsize, test_set = True, transform = transform)
        print(len(embedding_test))
        knn = KNeighborsClassifier(n_neighbors = 4, metric='manhattan')
        knn.fit(embedding_train, labels_train)
        preds_test = knn.predict(embedding_test)
        print(knn.score(embedding_test, labels_test))
        list_labels_cat = ['basophil', 'eosinophil', 'erythroblast','lymphocyte', 'monocyte', 'neutrophil']
        if source_test == ['rabin']:
            list_labels_cat = ['basophil', 'eosinophil','lymphocyte', 'monocyte', 'neutrophil']
        # plot_correlation(labels_test, preds_test, list_labels_cat)
        return preds_test
