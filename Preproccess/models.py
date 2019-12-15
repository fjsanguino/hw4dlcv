import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F


class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=2048):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        #set_parameter_requires_grad(self.resnet, feature_extracting=True)

        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        #cnn_embed_seq = []
        #for t in range(x_3d.size(0)):
            # ResNet CNN
        x = self.resnet(x_3d)#[:, t, :, :, :])  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc3(x)

            #cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        #cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return x #cnn_embed_seq

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
