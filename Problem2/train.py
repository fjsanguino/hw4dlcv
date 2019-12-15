import os
import torch

import parser
import models
import data
from reader import readShortVideo
from test1 import evaluate

from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms



from tensorboardX import SummaryWriter


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

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

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    feaStr_path = os.path.join(args.save_dir, 'cnn')
    if not os.path.exists(feaStr_path):
        os.makedirs(feaStr_path)

    class_path = os.path.join(args.save_dir, 'rnn')
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
                                               batch_size=args.train_batch,
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='valid'),
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=False)
    ''' load model '''
    print('===> prepare model ...')
    feature_stractor = models.ResCNNEncoder()
    feature_stractor.cuda()  # load model to gpu
    params_to_update = feature_stractor.parameters()
    params_to_update_str = []
    for name, param in feature_stractor.named_parameters():
        if param.requires_grad == True:
            params_to_update_str.append(param)

    rnn = models.DecoderRNN()
    rnn = rnn.cuda()
    params_to_update_rnn = []
    for name, param in rnn.named_parameters():
        if param.requires_grad == True:
            params_to_update_rnn.append(param)

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(params_to_update_rnn + params_to_update_str, lr=args.lr, weight_decay=args.weight_decay)


    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))
    iters = 0
    best_acc = 0
    loss_write = 0
    for epoch in range(1, args.epoch+1):
        rnn.train()
        feature_stractor.train()
        for idx, (video, video_path) in enumerate(train_loader):
            #max_dim = 0
            #for i in range(len(video_path)):
                #frames = readShortVideo(video_path[0], video.get('Video_category')[0], video.get('Video_name')[0])
                #max_dim = np.maximum(max_dim, frames.shape[0])
            #print(max_dim)


            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(train_loader))

            iters += 1

            #for i in range(len(video_path)):

            frames = readShortVideo(video_path[0], video.get('Video_category')[0], video.get('Video_name')[0])
            #print(frames.shape)
            vid = []
            for i in range(frames.shape[0]):
                im = transforms_array(frames[i])
                vid.append(im)
                #print(im.shape)
                #print(im)
            vid = torch.stack(vid)
            vid = torch.reshape(vid, (1, vid.shape[0], 3, 224, 224))
            #print(vid.shape)
            cls = (int(video.get('Action_labels')[0]))
            #print(cls)


            cls = torch.from_numpy(np.expand_dims(np.asarray(cls), axis=0))

            vid, cls  = vid.cuda(),  cls.cuda()

            _, pred = rnn(feature_stractor(vid))
            #print (pred.shape)




            #cls = int(video.get('Action_labels')[0])
            #cls = torch.from_numpy(np.expand_dims(np.asarray(cls), axis=0))
            #cls = cls.cuda()


            loss = criterion(pred, cls)

            #print('Back propagation')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #loss_write = loss_write + loss
            #if iters % 50 == 0:
            #    loss_write = loss_write/50
            #    writer.add_scalar('loss', loss_write.data.cpu().numpy(), iters)
            #    train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())
            #    print(train_info)
            #    loss_write = 0

            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)#, end = '\r')
        if epoch % args.val_epoch == 0:
            ''' evaluate the model '''
            acc = evaluate(feature_stractor, rnn, val_loader)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(feature_stractor, os.path.join(feaStr_path, 'model_best_cnn.pth.tar'))
                save_model(rnn, os.path.join(class_path, 'model_best_rnn.pth.tar'))
                best_acc = acc

        ''' save model '''
        save_model(feature_stractor, os.path.join(feaStr_path, 'model_{}_cnn.pth.tar'.format(epoch)))
        save_model(rnn, os.path.join(class_path, 'model_{}_rnn.pth.tar'.format(epoch)))


