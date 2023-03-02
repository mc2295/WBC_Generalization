import unittest
import torch
from main import Workflow
from unittest import TestCase
from models.siamese.siamese_models import SiameseModel
from models.encoder_head_model import Model, ModelFromResnet
from fastai.vision.all import Learner

exit_path = ''
device = 'cpu'
class TestWorkflow(unittest.TestCase):
    def test_create_dls_siamese(self):
        workflow = Workflow(exit_path)
        dls = workflow.create_dls(['test'], 8, siamese_head=True)
        b = dls.create_item(1)
        assert(len(b) == 3)

    def test_create_dls(self):
        workflow = Workflow(exit_path)
        dls = workflow.create_dls(['test'], 8, siamese_head=False)
        b = dls.create_item(1)
        assert(len(b) == 2)

    def test_create_model_siamese(self):
        workflow = Workflow(exit_path)
        model = workflow.create_model(siamese_head = True)
        assert isinstance(model, SiameseModel)

    def test_create_model_classifier(self):
        workflow = Workflow(exit_path)
        model = workflow.create_model(siamese_head = False)
        assert isinstance(model, ModelFromResnet)

    def test_create_learner_siamese(self):
        workflow = Workflow(exit_path)
        model = torch.load('models/test_siamese', map_location=device)
        dls = workflow.create_dls(['test'], 8, siamese_head=True)
        learn = workflow.create_learner(model, dls)
        assert isinstance(learn, Learner)

    def test_create_learner_classifer(self):
        workflow = Workflow(exit_path)
        model = torch.load('models/test_classifier', map_location=device)
        dls = workflow.create_dls(['test'], 8, siamese_head=False)
        learn = workflow.create_learner(model, dls)
        assert isinstance(learn, Learner)

    def test_train_model(self):
        workflow = Workflow(exit_path)    
        model = workflow.train_model(['test'], 8, 1, siamese_head = False, lr = 1e-2)  
        assert isinstance(model, ModelFromResnet)          

    def test_evaluate_model(self):
        workflow = Workflow(exit_path) 
        model = torch.load('models/test_classifier', map_location=device)
        accuracy = workflow.evaluate_model(model, ['test'],8)
        assert accuracy.item() <= 1

    def test_add_classifier(self):
        workflow = Workflow(exit_path)
        model = torch.load('models/test_siamese', map_location=device)
        model = workflow.add_classifier_head(model)
        assert isinstance(model, Model)

    def test_fine_tune(self):
        workflow = Workflow(exit_path)
        model = torch.load('models/test_siamese_classifier_head', map_location=device)
        model = workflow.fine_tune(model, ['test'], 8, 1, lr = 1e-2)
        assert isinstance(model, Model)
        
    def test_create_embeddings(self):
        workflow = Workflow(exit_path)
        model = torch.load('models/test_siamese', map_location=device)
        embedding, labels = workflow.create_embeddings(model, ['test'], 8)
        assert((embedding.shape == torch.Size([154, 256])) & (labels.shape == torch.Size([154])))

    def test_visualise_2D(self):
        workflow = Workflow(exit_path)
        model = torch.load('models/test_siamese', map_location=device)
        X_proj = workflow.visualise_2D(model, ['test'], 8, 't-SNE')
        assert X_proj.shape == torch.Size([154, 2])

    def test_knn(self):
        workflow = Workflow(exit_path)
        model = torch.load('models/test_classifier', map_location=device)
        pred = workflow.knn_on_embeddings(model, ['test'], ['test'], 8)
        assert pred.shape == torch.Size([154])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'])
