import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


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


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=2048, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=11):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, sequence, n_frames):
        packed = torch.nn.utils.rnn.pack_padded_sequence(sequence, n_frames, batch_first=True)
        RNN_out, (h_n, h_c) = self.LSTM(packed, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """
        #print(RNN_out[-1].shape)
        # FC layers
        x = self.fc1(h_n[-1])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        output = self.fc2(x)

        return x, output



