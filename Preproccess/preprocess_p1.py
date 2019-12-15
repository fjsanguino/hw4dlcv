import os
import torch

import parser
import models
import data
from reader import readShortVideo
import torchvision.transforms as transforms


import numpy as np

from tensorboardX import SummaryWriter

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



if __name__ == '__main__':

    args = parser.arg_parse()

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=False)
    val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='valid'),
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=False)
    ''' load model '''
    print('===> prepare model ...')
    #feature_stractor = models.ResCNNEncoder()
    feature_stractor = models.Stractor()
    feature_stractor = feature_stractor.cuda()  # load model to gpu

    train = 0
    if train == 1:
        features = []
        for idx, (video, video_path) in enumerate(train_loader):

            #print('working in batch', idx + 1)
            for i in range(len(video_path)):
                #print('working in video', i + 1, '/', idx + 1)
                frames = readShortVideo(video_path[i], video.get('Video_category')[i], video.get('Video_name')[i])
                print('working in video ', video.get('Video_index')[i], ' with size ', frames.shape)
                #print(frames.shape)
                vid = []
                for j in range(frames.shape[0]):
                    im = transforms_array(frames[j])
                    vid.append(im)
                    #print(im.shape)
                    #print(im)
                vid = torch.stack(vid)
                #vid = torch.reshape(vid, (1, vid.shape[0], 3, 224, 224))
                #print(vid.shape)
                vid = vid.cuda()
                feature = feature_stractor(vid)
                #print(feature.shape)
                feature = torch.mean(feature, 0)
                print(feature.shape)
                feature = feature.cpu().detach().numpy()
                features.append(feature)
        features = torch.tensor(features)
        print('features shape', features.shape)

        torch.save(features, 'train_p1.pt')

    validation = 1
    if validation == 1:
        features = []
        for idx, (video, video_path) in enumerate(val_loader):

            # print('working in batch', idx + 1)
            for i in range(len(video_path)):
                # print('working in video', i + 1, '/', idx + 1)
                frames = readShortVideo(video_path[i], video.get('Video_category')[i], video.get('Video_name')[i])
                print('working in video ', video.get('Video_index')[i], ' with size ', frames.shape)
                # print(frames.shape)
                vid = []
                for j in range(frames.shape[0]):
                    im = transforms_array(frames[j])
                    vid.append(im)
                    # print(im.shape)
                    # print(im)
                vid = torch.stack(vid)
                # vid = torch.reshape(vid, (1, vid.shape[0], 3, 224, 224))
                # print(vid.shape)
                vid = vid.cuda()
                feature = feature_stractor(vid)
                print(feature.shape)
                feature = torch.mean(feature, 0)
                print(feature.shape)
                feature = feature.cpu().detach().numpy()
                features.append(feature)
        features = torch.tensor(features)
        print('features shape', features.shape)

        torch.save(features, 'valid_p1.pt')

