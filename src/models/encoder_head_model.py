
import torch
import torch.nn as nn
from fastai.vision.all import Flatten
import torchvision
from transformers import ViTFeatureExtractor, ViTForImageClassification

'''
2 types of models:
ModelFromResnet:
- we add a linear layer at the end of the encoder
Model :
- we take 2 parts of a model (encoder, head) and make a full model, with siamese_head as parameter if siamese model
'''

class Encoder(nn.Module):
    def __init__(self, body, architecture):
        super(Encoder, self).__init__()
        # Define the layers for your encoder
        self.body = body
        self.architecture = architecture
        if architecture == 'inception' or architecture == 'efficientnet':
            self.linear_layer = nn.Linear(1000, 256)
        if architecture == 'resnet':
            self.linear_layer = nn.Sequential(nn.AdaptiveMaxPool2d(output_size=1),
            Flatten(),
            nn.Linear(2048, 256))
        if architecture == 'vgg':
            self.linear_layer = nn.Sequential(nn.AdaptiveMaxPool2d(output_size=1),
                            Flatten(),
                            nn.Linear(512, 256))
        if architecture == 'vit':
            self.linear_layer = nn.Linear(768, 256)
            self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

    def forward(self, x):
        if self.architecture == 'vit':
            model_layers = list(self.body.children())
            cut_model = model_layers[0]
            inputs = self.feature_extractor(images=x, return_tensors='pt')
            inputs = {key: value.to('cuda') for key, value in inputs.items()}
            outputs = cut_model(**inputs)
            sequence_output = outputs[0]
            out = self.linear_layer(sequence_output[:, 0, :])
            return out
        else:
            x = self.body(x)
            if isinstance(x, torchvision.models.inception.InceptionOutputs):
                out = self.linear_layer(x.logits)
                return out
            else:
                out = self.linear_layer(x)
                return out

class ModelFromArchitecture(nn.Module):
    def __init__(self,body, head, architecture):
        super(ModelFromArchitecture, self).__init__()
        self.body = body
        self.encoder = Encoder(body, architecture)
        self.head = head
        self.siamese_head = False
    def forward(self, x):
        out = self.encoder(x)
        return self.head(out)


class Model(torch.nn.Module):
    def __init__(self, encoder, head, siamese_head):
        super(Model, self).__init__()
        self.siamese_head = siamese_head
        self.encoder, self.head = encoder, head
    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)
