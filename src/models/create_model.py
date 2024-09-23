from efficientnet_pytorch import EfficientNet
from fastai.vision.all import xresnet101, create_body, create_head, models
from transformers import ViTForImageClassification
from models.encoder_head_model import ModelFromArchitecture
import torch


def create_model(architecture, n_out):
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
    head = create_head(128, n_out, ps=0.5)[2:] 
    return ModelFromArchitecture(body, head, architecture) 

