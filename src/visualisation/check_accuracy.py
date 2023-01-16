import numpy as np
import torch as torch


def treat(img, size=224):
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    return t.permute(2,0,1).float()/255.0

def check_accuracy(dls, model):
    list_avg = []
    for batch_index, batch_data in enumerate(dls.valid):
        results = model(batch_data[0].to('cpu'), batch_data[1].to('cpu'))
        label = results>0.5
        avg_batch = (label.squeeze(1) == batch_data[2].to('cpu')).float().mean()
        print(avg_batch)
        list_avg.append(avg_batch)
    moyenne = 0
    for k in list_avg:
        moyenne += k.item()
    return moyenne/len(list_avg)
