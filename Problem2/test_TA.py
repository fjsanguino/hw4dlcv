import models
import data
import parser
from reader import readShortVideo, getVideoList

import os
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

def load_model(args):
    rnn = models.DecoderRNN()

    feature = models.Stractor()

    model_dir_class = os.path.join(args.resume)

    model_std = torch.load(os.path.join(model_dir_class, 'model_best_rnn.pth.tar'), map_location="cuda:"+str(args.gpu))
    rnn.load_state_dict(model_std)
    rnn = rnn.cuda()

    model_std = torch.load(os.path.join(model_dir_class, 'model_best_fea.pth.tar'), map_location="cuda:"+str(args.gpu))
    feature.load_state_dict(model_std)
    feature = feature.cuda()
    #model = torch.nn.DataParallel(model,
                                  #device_ids=list(range(torch.cuda.device_count()))).cuda()

    return feature, rnn

def batch_padding(batch_fea):#, batch_cls):
    n_frames = [fea.shape[0] for fea in batch_fea]
    perm_index = np.argsort(n_frames)[::-1]

    # sort by sequence length
    batch_fea_sort = [batch_fea[i] for i in perm_index]
    #print(len(batch_fea_sort))
    n_frames = [fea.shape[0] for fea in batch_fea_sort]
    padded_sequence = nn.utils.rnn.pad_sequence(batch_fea_sort, batch_first=True)
    #label = torch.LongTensor(np.array(batch_cls)[perm_index])
    return padded_sequence, n_frames#label, n_frames




def store(feature_stractor, rnn, data_loader, batch_size):
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
            for i in range(len(video_path)):
                frames = readShortVideo(video_path[i], video.get('Video_category')[i], video.get('Video_name')[i])
                vid = []
                for j in range(frames.shape[0]):
                    im = transforms_array(frames[j])
                    vid.append(im)
                vid = torch.stack(vid).cuda()
                print('working in video ', video.get('Video_index')[i], ' with size ', vid.shape)
                feature = feature_stractor(vid)
                batch_img.append(feature)

            #print(batch_img[0].shape)
            #print(batch_img[1].shape)

            sequence, n_frames = batch_padding(batch_fea = batch_img)
            #print(sequence.shape)
            #print(n_frames)
            _, pred = rnn(sequence, n_frames)
            #print(pred.shape)

            _, pred = torch.max(pred, dim=1)


            pred = pred.cpu().numpy().squeeze()

            preds.append(pred)



        if batch_size != 1:
            preds = np.concatenate(preds)
    #print(preds.shape)
    print(preds)
    f = open("p2_result.txt", "w+")
    for pred in preds:
        f.write("%d\n" % pred)
    f.close()

if __name__ == '__main__':
    print('-----------------------------------------------Test-------------------------------------------------------')

    args = parser.arg_parse()

    json_dir = os.path.join(args.json_dir)


    torch.cuda.set_device(args.gpu)

    print("====================> Loading Data")
    val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='valid'),
                                                   batch_size=args.train_batch,
                                                   num_workers=args.workers,
                                                   shuffle=False)

    print("====================> Loading Model")
    feature, rnn = load_model(args)

    print("====================> Calculating the pred and writing the .txt")
    store(feature, rnn, val_loader, args.train_batch)

    with open('p2_result.txt') as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    pred = [x.strip() for x in content]
    pred = [int(x) for x in pred]
    pred = np.asarray(pred)


    data_dir = '../hw4_data'
    video_dir = os.path.join(data_dir, 'TrimmedVideos')

    ''' read the data list '''
    label_path = os.path.join(video_dir, 'label')
    label_path = os.path.join(label_path, 'gt_valid.csv')

    dic = getVideoList(label_path)
    gt = (dic.get('Action_labels'))
    gt = [int(x) for x in gt]
    gt = np.asarray(gt)
    print(accuracy_score(gt, pred))







