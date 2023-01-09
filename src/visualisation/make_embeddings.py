import numpy as np
from PIL import Image
import torch as torch
import torch.nn as nn

def open_image(fname, size=224):
    # print(fname)
    img = Image.open('../../Documents/These/' + fname).convert('RGB')
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    return t.permute(2,0,1).float()/255.0

def create_embeddings(Nimages, batch, valids, model):

    list_image = []
    print(len(valids))
    for i in range(Nimages*(batch-1), Nimages*batch):
        print(i, end = '\r')
        t1 = open_image(valids[i])
        list_image.append(model.encoder(t1.unsqueeze(0)))
    return list_image

def create_resdistance(Nimages, batch, valids, model):
    res = np.zeros((Nimages, Nimages))
    list_image = create_embeddings(Nimages, batch, valids, model)
    for i in range(Nimages):
        for j in range(i, Nimages):
            print(i, end = '\r')
            t1 = list_image[i]
            t2 = list_image[j]
            out = torch.abs(t1 - t2)
            out = model.head(out)
            out = nn.Sigmoid()(out)

            res[i][j] = out.item()
            res[j][i] = res[i][j]
    return res
