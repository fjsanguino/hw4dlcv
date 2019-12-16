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
import matplotlib.pyplot as plt



def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)



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
                                               shuffle=False)
    val_loader = torch.utils.data.DataLoader(data.DATA(args, mode='valid'),
                                             batch_size=args.train_batch,
                                             num_workers=args.workers,
                                             shuffle=False)
    ''' load model '''
    print('===> prepare model ...')

    classifier = models.Classifier()
    classifier = classifier.cuda()
    params_to_update_class = []

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))
    iters = 0
    best_acc = 0
    losses = []
    for epoch in range(1, args.epoch+1):
        classifier.train()
        for idx, (features, clss) in enumerate(train_loader):
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx + 1, len(train_loader))

            iters += 1

            features, clss = features.cuda(), clss.cuda()

            _, output = classifier(features)
            loss = criterion(output, clss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            losses.append(loss)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            #print(train_info)
        if epoch % args.val_epoch == 0:
            ''' evaluate the model '''
            acc = evaluate(classifier, val_loader)
            writer.add_scalar('val_acc', acc, iters)
            print('Epoch: [{}] ACC:{}'.format(epoch, acc))

            ''' save best model '''
            if acc > best_acc:
                save_model(classifier, os.path.join(args.save_dir, 'model_best_class.pth.tar'))
                best_acc = acc

        ''' save model '''
        save_model(classifier, os.path.join(args.save_dir, 'model_{}_class.pth.tar'.format(epoch)))

        plt.plot(range(len(losses)), losses)
        print(len(losses))
        if not os.path.exists('output'):
            os.makedirs('output')

        plt.savefig('output/train_loss.png')


