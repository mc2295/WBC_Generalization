import torch
from fastai.vision.all import params
import torch.nn as nn
# from fastai.vision.all import partial
# opt_func = partial(OptimWrapper, opt=optim.RMSprop)
# BCE_loss_f = partial(BCE_loss)

def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]

def CrossEnt_loss(out, targ):
    return CrossEntropyLossFlat()(out, targ.long())

def MCE_loss(out, target):
    res = (out - target).pow(2).mean()
    return res

def BCE_loss(out, target, reduction = 'mean'):
    return nn.BCELoss()(torch.squeeze(out, 1), target.float())

def contrastive_loss(y_pred, y_true):

    margin =1
    label = (y_pred > 0.5).squeeze(1).float()
    label = torch.tensor(label, requires_grad = True)
    square_pred = torch.square(label)
    a = torch.tensor(margin, dtype=torch.int8) - (label)
    b = torch.tensor(0, dtype = torch.int8).type_as(label)

    # margin_square = torch.square(torch.maximum(torch.tensor(margin, dtype=torch.int8) - (y_pred), torch.tensor(0, dtype=torch.int8)))
    margin_square = torch.square(torch.maximum(a, b))
    return torch.mean(
        (1 - y_true) * square_pred + (y_true) * margin_square
    )


def my_accuracy(input, target):
    label = input > 0.5
    return (label.squeeze(1) == target).float().mean()

def my_accuracy_2(output, target):
    euclidean_distance = F.pairwise_distance(output[0], output[1], keepdim = True)
    label = euclidean_distance > 0.5
    return (label.squeeze(1) == target).float().mean()


class ContrastiveLoss2(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss2, self).__init__()
        self.margin = margin

    def forward(self, output, label):

        # Calculate the euclidean distance and calculate the contrastive loss
        euclidean_distance = F.pairwise_distance(output[0], output[1], keepdim = True)
        print('label', label)

        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        print('distance', euclidean_distance)
        return loss_contrastive
