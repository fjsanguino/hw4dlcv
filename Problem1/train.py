import os
import torch

import parser
import models
import data
from reader import readShortVideo
from test1 import evaluate

import numpy as np
import torch.nn as nn
import torch.optim as optim


from tensorboardX import SummaryWriter


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)



if __name__ == '__main__':

    args = parser.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    feaStr_path = os.path.join(args.save_dir, 'featureStractor')
    if not os.path.exists(feaStr_path):
        os.makedirs(feaStr_path)

    class_path = os.path.join(args.save_dir, 'Classifier')
    if not os.path.exists(class_path):
        os.makedirs(class_path)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)

    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=32,
                                               num_workers=4,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='valid'),
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=False)
    ''' load model '''
    print('===> prepare model ...')
    feature_stractor = models.Stractor()
    feature_stractor.cuda()  # load model to gpu
    params_to_update = feature_stractor.parameters()
    params_to_update_str = []
    for name, param in feature_stractor.named_parameters():
        if param.requires_grad == True:
            params_to_update_str.append(param)

    classifier = models.Classifier()
    classifier = classifier.cuda()
    params_to_update_class = []
    for name, param in classifier.named_parameters():
        if param.requires_grad == True:
            params_to_update_class.append(param)

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(params_to_update_class + params_to_update_str, lr=args.lr, weight_decay=args.weight_decay)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))
    iters = 0
    best_acc = 0
    for epoch in range(1, args.epoch+1):
        classifier.train()
        feature_stractor.train()
        for idx, (video, video_path) in enumerate(train_loader):
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(train_loader))

            iters += 1

            features = []
            clss = []
            print('Preprocessing the data')
            for i in range(len(video_path)):
                #print('working ', i)
                frames = readShortVideo(video_path[i], video.get('Video_category')[i], video.get('Video_name')[i])
                frames_res = torch.from_numpy(frames)
                frames_res.resize_(len(frames), 3, 240, 240)
                frames_res = frames_res.float().cuda()
                print(feature_stractor(frames_res).shape, end="\r")
                features.append(torch.mean(feature_stractor(frames_res), 0).cpu().detach().numpy())
                clss.append(int(video.get('Action_labels')[i]))
            features = torch.from_numpy(np.asarray(features))
            clss = torch.from_numpy(np.asarray(clss))

            #FC
            print('Classifier')
            features, clss = features.cuda(), clss.cuda()

            _, output = classifier(features)
            loss = criterion(output, clss)

            print('Back propagation')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)
        if epoch % args.val_epoch == 0:
            ''' evaluate the model '''
            acc = evaluate(feature_stractor, classifier, val_loader)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(feature_stractor, os.path.join(feaStr_path, 'model_best_feaStr.pth.tar'))
                save_model(classifier, os.path.join(class_path, 'model_best_class.pth.tar'))
                best_acc = acc

        ''' save model '''
        save_model(feature_stractor, os.path.join(feaStr_path, 'model_{}_feaStr.pth.tar'.format(epoch)))
        save_model(classifier, os.path.join(class_path, 'model_{}_class.pth.tar'.format(epoch)))


