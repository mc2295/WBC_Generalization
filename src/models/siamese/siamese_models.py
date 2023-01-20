import torch.nn as nn
import torch
from torch.nn import Module
from fastai.vision.all import Flatten
import torchvision



class SiameseModel_gregoire(Module):
    def __init__(self, alpha_param = 0.01):


        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, padding = 1),
            nn.Conv2d(16, 16, kernel_size = 3, padding = 1),
            nn.MaxPool2d(kernel_size = 2),
            nn.Dropout(),


            nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
            nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
            nn.MaxPool2d(kernel_size = 2),
            nn.Dropout(),


            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.MaxPool2d(kernel_size = 2),
            nn.Dropout(),


            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.MaxPool2d(kernel_size = 2),
            nn.Dropout(),

            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.MaxPool2d(kernel_size = 2)
            # nn.AdaptiveMaxPool2d(512)

        )

        self.fc1 = nn.Linear(12544, 100)
        self.fc2 = nn.Linear(1,1)

        self.batchn = nn.BatchNorm1d(1)
        # self.fc = nn.Linear(512, 512)

        # self.encoder = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7, self.layer8, self.layer9, self.layer10])
        # self.encoder = nn.Sequential(self.encoder)
    def forward(self, x1, x2):
        x1 = self.cnn1(x1)
        x1 = x1.view(x1.size()[0], -1)
        x1 = self.fc1(x1)
        x2 = self.cnn1(x2)
        x2 = x2.view(x2.size()[0], -1)
        x2 = self.fc1(x2)

        out = F.pairwise_distance(x1, x2, keepdim = True)
        out = self.batchn(out)
        out = nn.Sigmoid()(out)
        return out

class SiameseModel0(Module):

    def __init__(self, encoder, head):
        super(SiameseModel, self).__init__()
        self.encoder,self.head = encoder,head

    # def forward(self, x1, x2):
    #     ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
    #     return self.head(ftrs)

    def similarity(self, x1, x2):
        x = torch.abs(x1 - x2)
        x = self.head(x)
        x = nn.Sigmoid()(x)
        return x

    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        return self.similarity(e1, e2)

class SiameseModel(Module):
    def __init__(self, resnet, head):
        super(SiameseModel, self).__init__()
        self.resnet, self.head = resnet, head
        self.encoder = nn.Sequential(
            self.resnet,
            nn.AdaptiveMaxPool2d(output_size=1),
            Flatten(),
            nn.Linear(2048, 256))

    def similarity(self, x1, x2):
#         x = F.pairwise_distance(x1, x2, keepdim = True)
        x = torch.abs(x1 - x2)        
        x = self.head(x)
        x = nn.Sigmoid()(x)
        return x

    def forward(self, x1, x2):
        e1 = self.encoder(x1)
        e2 = self.encoder(x2)
        return self.similarity(e1, e2)

class SiameseNetwork2(Module):

    def __init__(self):
        super(SiameseNetwork2, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(259584, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256,1)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        print(output.shape)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
