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

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


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
    feature_stractor = models.Stractor()
    feature_stractor = feature_stractor.cuda()  # load model to gpu
    feature_stractor.eval()

    train = 1
    if train == 1:
        features = []
        with torch.no_grad():
            for idx, (video, video_path) in enumerate(train_loader):
                for i in range(len(video_path)):
                    frames = readShortVideo(video_path[i], video.get('Video_category')[i], video.get('Video_name')[i])
                    vid = []
                    for j in range(frames.shape[0]):
                        im = transforms_array(frames[j])
                        vid.append(im)
                    vid = torch.stack(vid)
                    print('working in video ', video.get('Video_index')[i], ' with size ', vid.shape)
                    vid = vid.cuda()
                    feature = feature_stractor(vid)
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
        with torch.no_grad():
            for idx, (video, video_path) in enumerate(val_loader):
                for i in range(len(video_path)):
                    frames = readShortVideo(video_path[i], video.get('Video_category')[i], video.get('Video_name')[i])
                    print('working in video ', video.get('Video_index')[i], ' with size ', frames.shape)
                    vid = []
                    for j in range(frames.shape[0]):
                        im = transforms_array(frames[j])
                        vid.append(im)
                    vid = torch.stack(vid)
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

    save_model(feature_stractor, os.path.join(args.save_dir, 'model_best_fea.pth.tar'))
