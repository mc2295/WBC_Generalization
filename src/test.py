import unittest
import torch
from main import Workflow
from unittest import TestCase
import pandas as pd
from models.siamese.siamese_models import SiameseModel
from models.encoder_head_model import Model, ModelFromResnet
from fastai.vision.all import Learner

exit_path = ''
device = 'cpu'
class TestWorkflow(unittest.TestCase):
    def test_find_learning_rate(self):
        workflow = Workflow(exit_path)
        model = torch.load('models/test_classifier', map_location=device)
        lr = workflow.find_learning_rate(model, ['test'])
        assert(lr.valley<1)

    def test_train_model_siamese(self):
        workflow = Workflow(exit_path)
        model = workflow.train_model(['test'], 1, siamese_head=True, lr = 0.0001, batchsize = 8)
        assert isinstance(model, SiameseModel)

    def test_train_model(self):
        workflow = Workflow(exit_path)
        model = workflow.train_model(['test'], 1, siamese_head=False, lr = 0.0001,  batchsize = 8)
        assert isinstance(model, ModelFromResnet)

    def test_fine_tune_with_classifier_head(self):
        workflow = Workflow(exit_path)
        model = torch.load('models/test_siamese', map_location=device)
        model = workflow.fine_tune_with_classifier_head(model, ['test'], 8, 1, 0.0001)
        assert isinstance(model, Model)

    def test_visualise_datasets_intensity_2D(self):
        workflow = Workflow(exit_path)
        data = workflow.visualise_datasets_intensity_2D(10, ['test'])
        assert isinstance(data, pd.DataFrame)

    def test_visualise_embeddings_2D(self):
        workflow = Workflow(exit_path)
        model = torch.load('models/test_siamese', map_location=device)
        data = workflow.visualise_embeddings_2D(model, ['test'], 't-SNE')
        assert isinstance(data, pd.DataFrame)
    
    def test_display_images_from_scatter_plot_region(self):
        workflow = Workflow(exit_path)
        model = torch.load('models/test_classifier', map_location=device)
        df_embeddings = workflow.visualise_embeddings_2D(model, ['test'], 't-SNE')
        group = workflow.display_images_from_scatter_plot_region(['test'], model, df_embeddings, ['lymphocyte'], [-10, 10], [-10, 10])
        assert isinstance(group, pd.DataFrame)
    
    def test_show_confusion_matrix(self):
        workflow = Workflow(exit_path)
        model = torch.load('models/test_classifier', map_location=device)
        accuracy = workflow.show_confusion_matrix(model, ['test'])
        assert accuracy.item() <= 1

    def test_KNN(self):
        workflow = Workflow(exit_path)
        model = torch.load('models/test_classifier', map_location=device)
        pred = workflow.knn_on_embeddings(model, ['test'], ['test'], 8)
        assert pred.shape == torch.Size([37])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'])
