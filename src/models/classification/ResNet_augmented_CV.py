# from pyimagesearch import config
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import torch
from torch.utils.tensorboard import SummaryWriter
import csv
import numpy as np
from numpy import vstack, floor
from numpy import argmax, shape
from pandas import read_csv
from sklearn.metrics import accuracy_score
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader, random_split
import random
import torch.nn as nn
# import torch.hub as hub
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import timm
from os import listdir
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.version.cuda)





# For unbalanced dataset we create a weighted sampler

def train_model(train_dl, valid_dl, model, num_epochs=20, criterion= CrossEntropyLoss()):
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    total_step = len(train_dl)



    for epoch in range(num_epochs):
        correct_train = 0
        total_train = 0

        acc_train = [0 for c in range(21)]
        total_per_class_train = [0 for c in range(21)]

        for i, (images, labels) in enumerate(train_dl):
                # Move tensors to the configured device

            if i%100 == 0:
                with open('ResNextlog_CV.txt', 'a') as f:
                    print('Batch ' + str(i))
                    f.write('Batch ' + str(i) + '\n')
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)


            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs.data, 1)

            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()

            # print(labels, preds)
            for c in range(21):
                acc_train[c] +=((preds == labels) * (labels == c)).sum().item()
                total_per_class_train[c] += (labels == c).sum().item()
        with open('results_ResNext_augmented_CV.csv', 'a') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([epoch, loss.item(), 100 * correct_train / total_train, '', [np.round(100 * acc_train[c]/max(total_per_class_train[c],1),decimals=1) for c in range(21)]])

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct_train / total_train))
        print('Accuracy per class', [np.round(100 * acc_train[c]/max(total_per_class_train[c],1),decimals=1) for c in range(21)])


        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            index_val = 0
            acc = [0 for c in range(21)]
            total_per_class = [0 for c in range(21)]
            for images, labels in valid_dl:
                index_val+=1
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                # _, predicted = torch.max(outputs.data, 1)
                # del images, labels, outputs


                _, preds = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

                # print(labels, preds)
                for c in range(21):
                    acc[c] +=((preds == labels) * (labels == c)).sum().item()
                    total_per_class[c] += (labels == c).sum().item()
                del images, labels, outputs

            with open('results_ResNext_augmented_CV.csv', 'a') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([epoch, loss.item(),'' , 100 * correct / total,[np.round(100 * acc[c]/max(total_per_class[c],1),decimals=1) for c in range(21)]])

            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))
            print('Accuracy per class', [np.round(100 * acc[c]/max(total_per_class[c],1),decimals=1) for c in range(21)])

def train_with_CV(path_train, path_test, model0, batch_size = 32):


        hFlip = transforms.RandomHorizontalFlip(p=0.25)
        vFlip = transforms.RandomVerticalFlip(p=0.25)
        rotate = transforms.RandomRotation(degrees=15)
        color = transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=0,hue=(0,0.1))

        trainTransforms = transforms.Compose([ hFlip, vFlip, rotate, color,
                transforms.ToTensor()])

        def make_weights_for_balanced_classes(images, nclasses):
                count = [0] * nclasses
                for item in images:
                        count[item[1]] += 1
                weight_per_class = [0.] * nclasses
                N = float(sum(count))
                for i in range(nclasses):
                        weight_per_class[i] = N/float(count[i])
                weight = [0] * len(images)
                for idx, val in enumerate(images):
                        weight[idx] = weight_per_class[val[1]]
                return weight


        Dataset1 = ImageFolder(root=path_train,
                transform= trainTransforms)
        Dataset2 = ImageFolder(root=path_test,
                transform=transforms.ToTensor())

        allDataset = torch.utils.data.ConcatDataset([Dataset1, Dataset2])
        seed = random.randint(1,100)
        prior_weights1, prior_weights2 = make_weights_for_balanced_classes(Dataset1.imgs, len(Dataset1.classes)), make_weights_for_balanced_classes(Dataset2.imgs, len(Dataset2.classes))
        weights = prior_weights1 + prior_weights2
        weights = torch.DoubleTensor(weights)
        splits=KFold(n_splits=5,shuffle=True,random_state=seed)

        for fold,(train_idx,test_idx) in enumerate(splits.split(np.arange(len(allDataset)))):
                model = model0
                model = model.to(DEVICE)
                print('------------fold no---------{}----------------------'.format(fold))

                weights1, weights2 = weights[train_idx], weights[test_idx]
                train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights1, len(weights1))
                test_sampler =torch.utils.data.SubsetRandomSampler(test_idx)
                train_dl = torch.utils.data.DataLoader(
                                        allDataset,
                                        batch_size=batch_size, sampler=train_sampler)
                valid_dl= torch.utils.data.DataLoader(
                                        allDataset,
                                        batch_size=batch_size, sampler=test_sampler)
                train_model(train_dl, valid_dl, model, num_epochs=15, criterion= CrossEntropyLoss())

with open('results_ResNext_augmented_CV.csv', 'w') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(['Epoch', 'Loss', 'Train precision', 'Val precision', 'Precision per class'])
with open('ResNextlog_CV.txt', 'w') as f:
    f.write('Debut ' + str(torch.cuda.is_available()))

path_train = '../../local/manon/train'
path_test = '../../local/manon/test'
model = timm.create_model('resnext101_32x8d', pretrained=True)
# torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
train_with_CV(path_train, path_test, model)


torch.save(model, 'models/ResNext_model_augmented_CV')
