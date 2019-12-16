import os
import json
import torch
import scipy.misc

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

from reader import getVideoList, readShortVideo

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.video_dir = os.path.join(self.data_dir, 'TrimmedVideos')

        ''' read the data list '''
        self.label_path = os.path.join(self.video_dir, 'label')
        self.label_path = os.path.join(self.label_path, 'gt_' + mode + '_less.csv')

        self.video_path = os.path.join(self.video_dir, 'video')
        self.video_path = os.path.join(self.video_path, mode)

        self.dic = getVideoList(self.label_path)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            transforms.Normalize(MEAN, STD)
        ])


    def __len__(self):
        return len(self.dic.get('Video_index'))

    def __getitem__(self, idx):

        video = {}
        for x, y in self.dic.items():
            video[x] = y[idx]
        ''' get data '''
        #frames = readShortVideo(self.video_path, video.get('Video_category'), video.get('Video_name'))


        #cls = video.get('Video_index')
        ''' read image '''


        return video, self.video_path