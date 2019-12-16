import torch
import data
import parser
import models
from reader import readShortVideo

import os
from sklearn.manifold import TSNE
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import torchvision.transforms as transforms
import torch.nn as nn


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


def output_features(rnn, feature_stractor, data_loader, json_dir):
    ''' set model to evaluate mode '''
    rnn.eval()
    feature_stractor.eval()
    iters  = 0
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        data = []
        for idx, (video, video_path) in enumerate(data_loader):

            print(iters)
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
            # print(sequence.shape)

            feat, _ = rnn(sequence, n_frames)

            features_flt = []
            for imgs in feat:
                imgs_feature = []
                for fea in imgs:
                    imgs_feature.append(float(fea))
                features_flt.append(list(imgs_feature))

            ##strore the values of the pred.

            for i in range(0, len(features_flt)):
                data.append([list(features_flt[i]), batch_gt[i]])

        data = list(data)
        with open(json_dir, 'w') as outfile:
            json.dump(data, outfile)



def load_model(args):
    rnn = models.DecoderRNN()

    model_dir_rnn = args.resume

    model_std = torch.load(os.path.join(model_dir_rnn, 'model_best_rnn.pth.tar'), map_location="cuda:"+str(args.gpu))
    rnn.load_state_dict(model_std)
    rnn = rnn.cuda()

    feature_stractor = models.Stractor()
    feature_stractor = feature_stractor.cuda()


    #model = torch.nn.DataParallel(model,
                                  #device_ids=list(range(torch.cuda.device_count()))).cuda()

    return feature_stractor, rnn

if __name__ == '__main__':
    print('-----------------------------------------------Representation-------------------------------------------------------')


    plt.title("Simple Plot")
    args = parser.arg_parse()

    json_dir = os.path.join(args.json_dir)

    if (not os.path.exists(os.path.join(json_dir, 'data.json'))):

        torch.cuda.set_device(args.gpu)

        print("====================> Loading Data")
        val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='valid'),
                                                 batch_size=args.train_batch,
                                                 num_workers=args.workers,
                                                 shuffle=False)

        print("====================> Loading Model")
        feature_stractor, rnn = load_model(args)

        print("====================> Calculating the features and writing the json")
        output_features(rnn, feature_stractor, val_loader, os.path.join(json_dir, 'features.json'))

    print("====================> Reading the jsons")
    #always read the data and plot it
    with open(os.path.join(json_dir, 'features.json'), 'r') as f:
        data = json.load(f)


    fea=[]
    cls=[]
    for d in data:
        fea.append(d[0])
        cls.append(d[1])

    fea_test = []
    for i in range(0, 5):
        fea_test.append(fea[i])

    print("====================> Calculating TSNE")

    fea_embedded = TSNE(n_components=2, n_iter=300).fit_transform(fea)

    print("====================> Calculating the colors")
    clr = []
    for i in range(0, len(cls)):
        if (cls[i] == 0): clr.append('r')
        if (cls[i] == 1): clr.append('b')
        if (cls[i] == 2): clr.append('g')
        if (cls[i] == 3): clr.append('c')
        if (cls[i] == 4): clr.append('m')
        if (cls[i] == 5): clr.append('y')
        if (cls[i] == 6): clr.append('k')
        if (cls[i] == 7): clr.append('pink')
        if (cls[i] == 8): clr.append('orange')
        if (cls[i] == 9): clr.append('slategray')
        if (cls[i] == 10): clr.append('lawngreen')


    print("====================> Painting data")

    for i in range(0, len(fea_embedded)):
        plt.plot(fea_embedded[i][0], fea_embedded[i][1], clr[i], marker='o')

    if not os.path.exists('output'):
        os.makedirs('output')

    plt.savefig('output/representation_class.png')

