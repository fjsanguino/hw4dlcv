import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms



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
