import numpy as np
import torch as torch


def treat(img, size=224):
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    return t.permute(2,0,1).float()/255.0

def check_accuracy(tls, model, nb_samples):
    correct = 0
    for k in range(nb_samples):

        im1 = treat(tls.valid[k][0])
        im2 = treat(tls.valid[k][1])
        result = model(im1.unsqueeze(0), im2.unsqueeze(0))
        print(k,  end = '\r')
        if int(np.round(result.item()))== tls[k][2]:
            correct+=1
    return correct
