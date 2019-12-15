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

def batch_padding(batch_fea, batch_cls):
    n_frames = [fea.shape[0] for fea in batch_fea]
    perm_index = np.argsort(n_frames)[::-1]

    # sort by sequence length
    batch_fea_sort = [batch_fea[i] for i in perm_index]
    #print(len(batch_fea_sort))
    n_frames = [fea.shape[0] for fea in batch_fea_sort]
    padded_sequence = nn.utils.rnn.pad_sequence(batch_fea_sort, batch_first=True)
    label = torch.LongTensor(np.array(batch_cls)[perm_index])
    return padded_sequence, label, n_frames



def evaluate(feature_stractor, rnn, data_loader, batch_size):
    ''' set model to evaluate mode '''
    rnn.eval()
    feature_stractor.eval()
    iters = 0
    gts = []
    preds = []
    with torch.no_grad():
        for idx, (video, video_path) in enumerate(data_loader):
            #print(iters)
            iters += 1
            batch_img = []
            batch_gt = []
            for i in range(len(video_path)):
                frames = readShortVideo(video_path[i], video.get('Video_category')[i], video.get('Video_name')[i])

                vid = []
                for j in range(frames.shape[0]):
                    im = transforms_array(frames[j])
                    vid.append(im)
                vid = torch.stack(vid).cuda()

                with torch.no_grad():
                    feature = feature_stractor(vid)

                batch_img.append(feature)

                gt = (int(video.get('Action_labels')[i]))
                batch_gt.append(gt)


            sequence, label, n_frames = batch_padding(batch_img, batch_gt)
            #print(sequence.shape)

            _, pred = rnn(sequence, n_frames)

            _, pred = torch.max(pred, dim=1)

            batch_gt = torch.from_numpy(np.asarray(batch_gt))
            # print(batch_gt.shape)

            pred = pred.cpu().numpy().squeeze()
            batch_gt = batch_gt.numpy().squeeze()

            preds.append(pred)
            gts.append(batch_gt)



        if batch_size != 1:
            gts = np.concatenate(gts)
            preds = np.concatenate(preds)
    print(preds)
    return accuracy_score(gts, preds)

