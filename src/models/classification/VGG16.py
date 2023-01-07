# from pyimagesearch import config
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
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
from os import listdir
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(torch.version.cuda)
def visualize_batch(batch, classes, dataset_type):
	# initialize a figure
	fig = plt.figure("{} batch".format(dataset_type),
		figsize=(batch_size, batch_size))
	# loop over the batch size
	for i in range(0, batch_size):
		# create a subplot
		ax = plt.subplot(2, 4, i + 1)
		# grab the image, convert it from channels first ordering to
		# channels last ordering, and scale the raw pixel intensities
		# to the range [0, 255]
		image = batch[0][i].cpu().numpy()
		image = image.transpose((1, 2, 0))
		image = (image * 255.0).astype("uint8")
		# grab the label id and get the label from the classes list
		idx = batch[1][i]
		label = classes[idx]
		# show the image along with the label
		plt.imshow(image)
		plt.title(label)
		plt.axis("off")
	# show the plot
	plt.tight_layout()
	plt.show()

class VGG16(nn.Module):
    def __init__(self, num_classes=21):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out



def prepare_data(path_train, path_test, batch_size=32, split_size=0.8):

        # define standardization
        # trans = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

        ## Data augmentation

        hFlip = transforms.RandomHorizontalFlip(p=0.25)
        vFlip = transforms.RandomVerticalFlip(p=0.25)
        rotate = transforms.RandomRotation(degrees=15)

        # initialize our training and validation set data augmentation
        # pipeline
        trainTransforms = transforms.Compose([ hFlip, vFlip, rotate,
        transforms.ToTensor()])
        valTransforms = transforms.Compose([transforms.ToTensor()])

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




        trainDataset = ImageFolder(root=path_train,
                transform=transforms.ToTensor())

        seed = random.randint(1,100)
        weights = make_weights_for_balanced_classes(trainDataset.imgs, len(trainDataset.classes))


        n_train = int(floor(split_size * len(trainDataset)))
        n_val = len(trainDataset) - n_train
        train_ds, val_ds = random_split(trainDataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

        weights = torch.DoubleTensor(weights)
        weights1, weight2 = random_split(weights, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights1, len(weights1))

        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler = sampler)
        # train_dl = DataLoader(train_ds, batch_size=batch_size)
        valid_dl = DataLoader(val_ds , batch_size=batch_size)


        testDataset = ImageFolder(root=path_test,
                transform=transforms.ToTensor())
        test_dl = DataLoader(testDataset, batch_size=batch_size)

        return train_dl, valid_dl, test_dl






# For unbalanced dataset we create a weighted sampler


def train_model(train_dl, valid_dl, model, num_epochs=10, criterion= CrossEntropyLoss()):
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    total_step = len(train_dl)

    with open('results_VGG16.csv', 'w') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Epoch', 'Loss', 'Train acc', 'Train acc per class', 'Val acc', 'Val acc per class'])
    with open('VGG16log.txt', 'w') as f:
        f.write('Debut ' + str(torch.cuda.is_available()))
    for epoch in range(num_epochs):
        correct_train = 0
        total_train = 0

        acc_train = [0 for c in range(21)]
        total_per_class_train = [0 for c in range(21)]

        for i, (images, labels) in enumerate(train_dl):
                    # Move tensors to the configured device
            if i%100 == 0:
                with open('VGG16log.txt', 'a') as f:
                    print('Batch ' + str(i))
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
        with open('results_VGG16.csv', 'a') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([epoch, loss.item(), 100 * correct_train / total_train, [np.round(100 * acc_train[c]/max(total_per_class_train[c],1),decimals=1) for c in range(21)], '',''])

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        with open('VGG16log.txt', 'a') as f:
            f.write('Epoch ' + str(epoch))
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

            with open('results_VGG16.csv', 'a') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(['', loss.item(),'' , '',  100 * correct / total, [np.round(100 * acc[c]/max(total_per_class[c],1),decimals=1) for c in range(21)]])

            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))
            print('Accuracy per class', [np.round(100 * acc[c]/max(total_per_class[c],1),decimals=1) for c in range(21)])
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):

        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # convert to class labels
        yhat = argmax(yhat, axis=1)
        # reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    return acc



path_train = '../../local/manon/train'
path_test = '../../local/manon/test'
labels_unique = listdir(path_train)


num_classes = 4
num_epochs = 2
batch_size = 32
learning_rate = 0.005

train_dl, valid_dl, test_dl = prepare_data(path_train, path_test)

total_train = len(train_dl.dataset)
# print(total_train)
class_weights = [len(listdir(path_train + "/" + i + "/"))/total_train for i in listdir(path_train)]

# print(len(train_dl.dataset), len(test_dl.dataset))
model = VGG16()
model = model.to(DEVICE)


train_model(train_dl, valid_dl, model)
torch.save(model, 'models/VGG16_model')
