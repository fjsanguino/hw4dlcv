import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

class Stractor(nn.Module):

    def __init__(self):
        super(Stractor, self).__init__()

        ''' declare layers used in this network'''
        # first layer: resnet18 gets a pretrained model
        self.resnet50 = models.resnet50(pretrained=True)
        #        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, 2048)
        # or instead of the above function you can use:
        # self.resnet18.fc = Identity()
        # self.resnet18.avgPool = Identity()



    def forward(self, img):
        x = self.resnet50(img)


        return x


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 11)

    def forward(self, feat):
        x = self.fc1(feat)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        output = self.fc6(x)


        return x, output
