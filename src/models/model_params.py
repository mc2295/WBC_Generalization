from fastai.vision.all import L, params
def  split_layers(m):
    return L(m.encoder, m.head).map(params)
