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


def output_features(classi, feaStract, data_loader, json_dir):
    ''' set model to evaluate mode '''
    classi.eval()
    feaStract.eval()
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        data = []
        for idx, (video, video_path) in enumerate(data_loader):
            features = []
            clss = []
            print('Preprocessing the data')
            for i in range(len(video_path)):
                # print('working ', i)
                frames = readShortVideo(video_path[i], video.get('Video_category')[i], video.get('Video_name')[i])
                frames_res = torch.from_numpy(frames)
                frames_res.resize_(len(frames), 3, 240, 240)
                frames_res = frames_res.float().cuda()
                # print(torch.mean(feature_stractor(frames_res), 0).shape)
                features.append(torch.mean(feaStract(frames_res), 0).cpu().detach().numpy())
                clss.append(int(video.get('Action_labels')[i]))
            features = torch.from_numpy(np.asarray(features))
            #clss = torch.from_numpy(np.asarray(clss))

            # FC
            print('Classifier')
            features = features.cuda()

            feat, _ = classi(features)
            features_flt = []
            for imgs in feat:
                imgs_feature = []
                for fea in imgs:
                    imgs_feature.append(float(fea))
                features_flt.append(list(imgs_feature))

            ##strore the values of the pred.

            for i in range(0, len(features_flt)):
                data.append([list(features_flt[i]), clss[i]])

        data = list(data)
        with open(json_dir, 'w') as outfile:
            json.dump(data, outfile)



def load_model(args):
    classifier = models.Classifier()
    stract = models.Stractor()

    model_dir_class = os.path.join(args.resume, 'Classifier')

    model_dir_str = os.path.join(args.resume, 'featureStractor')

    model_std = torch.load(os.path.join(model_dir_class, 'model_1_class.pth.tar'), map_location="cuda:"+str(args.gpu))
    classifier.load_state_dict(model_std)
    classifier = classifier.cuda()

    model_std = torch.load(os.path.join(model_dir_str, 'model_1_feaStr.pth.tar'), map_location="cuda:"+str(args.gpu))
    stract.load_state_dict(model_std)
    stract = stract.cuda()


    #model = torch.nn.DataParallel(model,
                                  #device_ids=list(range(torch.cuda.device_count()))).cuda()

    return stract, classifier

if __name__ == '__main__':


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
        feaStractor, classifier = load_model(args)

        print("====================> Calculating the features and writing the json")
        output_features(classifier, feaStractor, val_loader, os.path.join(json_dir, 'features.json'))

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
    for i in range(0, 10):
        fea_test.append(fea[i])

    print("====================> Calculating TSNE")

    fea_embedded = TSNE(n_components=2, n_iter=300).fit_transform(fea_test)

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

