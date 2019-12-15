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


def output_features(classi, data_loader, json_dir):
    ''' set model to evaluate mode '''
    classi.eval()
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        data = []
        with torch.no_grad():  # do not need to caculate information for gradient during eval
            for idx, (features, gt) in enumerate(data_loader):
                features = features.cuda()
                # print(features.shape)
                # print(gt.shape)

                feat, _ = classi(features)
                #print(feat.shape)
                #print(gt.shape)
                features_flt = []
                for imgs in feat:
                    imgs_feature = []
                    for fea in imgs:
                        imgs_feature.append(float(fea))
                    features_flt.append(list(imgs_feature))
                #print(np.asarray(features_flt).shape)
                clss = []
                for cls in gt:
                    clss.append(float(cls))
                print(np.asarray(clss).shape)


                ##strore the values of the pred.

                for i in range(0, len(features_flt)):
                    data.append([list(features_flt[i]), clss[i]])

            data = list(data)
            with open(json_dir, 'w') as outfile:
                json.dump(data, outfile)


def load_model(args):
    classifier = models.Classifier()

    model_dir_class = os.path.join(args.resume)



    model_std = torch.load(os.path.join(model_dir_class, 'model_best_class.pth.tar'), map_location="cuda:"+str(args.gpu))
    classifier.load_state_dict(model_std)
    classifier = classifier.cuda()


    #model = torch.nn.DataParallel(model,
                                  #device_ids=list(range(torch.cuda.device_count()))).cuda()

    return classifier

if __name__ == '__main__':


    plt.title("Simple Plot")
    args = parser.arg_parse()

    json_dir = os.path.join(args.json_dir)

    if (not os.path.exists(os.path.join(json_dir, 'features.json'))):

        torch.cuda.set_device(args.gpu)

        print("====================> Loading Data")
        val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='valid'),
                                                   batch_size=args.train_batch,
                                                   num_workers=args.workers,
                                                   shuffle=True)

        print("====================> Loading Model")
        classifier = load_model(args)

        print("====================> Calculating the features and writing the json")
        output_features(classifier, val_loader, os.path.join(json_dir, 'features.json'))

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
        print(i)
        plt.plot(fea_embedded[i][0], fea_embedded[i][1], clr[i], marker='o')

    if not os.path.exists('output'):
        os.makedirs('output')

    plt.savefig('output/representation_class.png')

