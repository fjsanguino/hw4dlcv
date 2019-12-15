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

if __name__ == '__main__':

    args = parser.arg_parse()

    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


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
    feature_stractor = models.Stractor()
    feature_stractor = feature_stractor.cuda()  # load model to gpu

    rnn = models.DecoderRNN()
    rnn = rnn.cuda()

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))
    iters = 0
    best_acc = 0
    loss_write = 0
    for epoch in range(1, args.epoch+1):
        rnn.train()
        feature_stractor.eval()
        for idx, (video, video_path) in enumerate(train_loader):


            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(train_loader))

            iters += 1
            batch_img = []
            batch_cls = []
            for i in range(len(video_path)):
                #print(i)
                frames = readShortVideo(video_path[i], video.get('Video_category')[i], video.get('Video_name')[i])
                #print(video.get('Video_index')[i])
                #print('frames shape', frames.shape)
                vid = []
                for j in range(frames.shape[0]):
                    im = transforms_array(frames[j])
                    vid.append(im)
                vid = torch.stack(vid).cuda()
                print('video shape', vid.shape, end='\r')
                #vid = torch.reshape(vid, (1, vid.shape[0], 3, 224, 224))
                #print(vid.shape)
                with torch.no_grad():
                    feature = feature_stractor(vid)
                #print(feature.shape)
                batch_img.append(feature)

                cls = (int(video.get('Action_labels')[i]))
                batch_cls.append(cls)

            #print((batch_cls[0].type))
            #print(len(batch_img))

            sequence, label, n_frames = batch_padding(batch_img, batch_cls)
            print('sequence shape', sequence.shape,  end='\r')

            _, pred = rnn(sequence, n_frames)
            #print(pred.shape)

            #print(batch_cls)
            batch_cls = torch.from_numpy(np.asarray(batch_cls)).cuda()
            #print(batch_cls.shape)



            #cls = int(video.get('Action_labels')[0])
            #cls = torch.from_numpy(np.expand_dims(np.asarray(cls), axis=0))
            #cls = cls.cuda()


            loss = criterion(pred, batch_cls)

            #print('Back propagation')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)#, end = '\r')
        if epoch % args.val_epoch == 0:
            ''' evaluate the model '''
            acc = evaluate(feature_stractor, rnn, val_loader, args.train_batch)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(rnn, os.path.join(args.save_dir, 'model_best_rnn.pth.tar'))
                best_acc = acc

        ''' save model '''
        save_model(rnn, os.path.join(args.save_dir, 'model_{}_rnn.pth.tar'.format(epoch)))


