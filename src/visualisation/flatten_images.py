import torch
import numpy as np

def create_matrix_of_flattened_images(valid_loader):
    flattened_vector = []
    datasets = []
    
    for imgs, _, _, dataset in valid_loader:
        image_size = imgs.shape[-1]
        flattened_vector += [torch.sum(k, dim = 0).view(image_size*image_size) for k in imgs]
        datasets += [k for k in dataset]
    return torch.stack(flattened_vector), datasets
