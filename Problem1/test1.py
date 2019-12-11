import models
import data

from reader import readShortVideo
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score


def evaluate(feature_stractor, classifier, data_loader):
    ''' set model to evaluate mode '''
    feature_stractor.eval()
    classifier.eval()
    preds = []
    gts = []
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (video, video_path) in enumerate(data_loader):
            features = []
            gt = []
            print('Preprocessing the data')
            for i in range(len(video_path)):
                # print('working ', i)
                frames = readShortVideo(video_path[i], video.get('Video_category')[i], video.get('Video_name')[i])
                frames_res = torch.from_numpy(frames)
                frames_res.resize_(len(frames), 3, 240, 240)
                frames_res = frames_res.float().cuda()
                # print(torch.mean(feature_stractor(frames_res), 0).shape)
                features.append(torch.mean(feature_stractor(frames_res), 0).cpu().detach().numpy())
                gt.append(int(video.get('Action_labels')[i]))
            features = torch.from_numpy(np.asarray(features))

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
