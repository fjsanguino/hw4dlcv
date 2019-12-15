import models
import data

from reader import readShortVideo
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score


def evaluate(classifier, data_loader):
    ''' set model to evaluate mode '''
    classifier.eval()
    preds = []
    gts = []
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (features, gt) in enumerate(data_loader):

            features, gt = features.cuda(), gt.cuda()

            _, pred = classifier(features)

            _, pred = torch.max(pred, dim=1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()

            preds.append(pred)
            gts.append(gt)

        gts = np.concatenate(gts)
        preds = np.concatenate(preds)

        return accuracy_score(gts, preds)
