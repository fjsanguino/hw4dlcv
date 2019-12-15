import models
import data

from reader import readShortVideo
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms


from sklearn.metrics import accuracy_score

def transforms_array(array):
    MEAN = [0.5, 0.5, 0.5]
    STD = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.ToPILImage(mode='RGB'),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
        transforms.Normalize(MEAN, STD)
    ])
    return transform(array)



def evaluate(feature_stractor, rnn, data_loader):
    ''' set model to evaluate mode '''
    feature_stractor.eval()
    rnn.eval()
    preds = []
    gts = []
    i = 0
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (video, video_path) in enumerate(data_loader):

            frames = readShortVideo(video_path[0], video.get('Video_category')[0], video.get('Video_name')[0])
            # print(frames.shape)
            vid = []
            for i in range(frames.shape[0]):
                im = transforms_array(frames[i])
                vid.append(im)
            vid = torch.stack(vid)
            vid = torch.reshape(vid, (1, vid.shape[0], 3, 224, 224))

            vid = vid.cuda()

            _, pred = rnn(feature_stractor(vid))
            _, pred = torch.max(pred, dim=1)

            gt = int(video.get('Action_labels')[0])
            gt = torch.from_numpy(np.expand_dims(np.asarray(gt), axis=0))

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()

            preds.append(pred)
            gts.append(gt)
            #print(preds)
            #print(gts)



        #gts = np.concatenate(gts)
        #preds = np.concatenate(preds)
    print(preds)
    return accuracy_score(gts, preds)

