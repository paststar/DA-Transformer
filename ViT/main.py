"""
TODO
 1. Loss/Accuracy visualize
 2. Knowledge Distillation
 3. Augumentation (mixup)
 4. Pretrained? <- 가능 한가?
 5. Multi source로 확장
"""

import os
import copy
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata
from torch.utils.tensorboard import SummaryWriter

from train import train_transformer
from dataset import get_dataset
import utils as utils
from model import *

parser = argparse.ArgumentParser(description="DANN EXPERIMENTS")
parser.add_argument('-db_path', help='gpu number', type=str, default='../Dataset')
parser.add_argument('-baseline_path', help='baseline path', type=str, default='AD_Baseline')
#parser.add_argument('-save_path', help='save path', type=str, default='Logs/save_test')

# amazon, webcam, dslr
parser.add_argument('-source', help='source', type=str, default='dslr')
parser.add_argument('-target', help='target', type=str, default='webcam')
parser.add_argument('-experiment_name', help='experiment_name', type=str, default='DeiT_tiny')

parser.add_argument('-workers', default=4, type=int, help='dataloader workers')
parser.add_argument('-gpu', help='gpu number', type=str, default='0')

parser.add_argument('-epochs', default=200, type=int)
parser.add_argument('-batch_size', default=32, type=int)
parser.add_argument('-lr', default=0.001, type=float)
parser.add_argument('-l2_decay', default=0.01, type=float)
parser.add_argument('-momentum', default=0.9, type=float)
parser.add_argument('-nesterov', default=False, type=bool)

def main():
    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("Use GPU(s): {} for training".format(args.gpu))

    save_dir = os.path.join("./logs", args.experiment_name+" "+args.source[0].upper() + ' -> ' + args.target[0].upper() + datetime.now().strftime('\n%m%d\n%H:%M'))
    writer = SummaryWriter(save_dir)
    #if os.path.exists(save_dir):
        #raise NameError('model dir exists!')
    #os.makedirs(save_dir)
    logging = utils.init_log(save_dir)
    _print = logging.info
    _print("############### experiments setting ###############")
    _print(args.__dict__)
    _print("###################################################")

    num_classes, resnet_type = utils.get_data_info()
    src_trainset, src_testset = get_dataset(args.source, path=args.db_path)
    tgt_trainset, tgt_testset = get_dataset(args.target, path=args.db_path)

    src_train_loader = torchdata.DataLoader(src_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    tgt_train_loader = torchdata.DataLoader(tgt_trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    tgt_test_loader = torchdata.DataLoader(tgt_testset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=False)

    lr, l2_decay, momentum, nesterov = args.lr, args.l2_decay, args.momentum, args.nesterov


    #transformer = ViT.ViT().cuda()
    transformer = DeiT_tiny().cuda()

    optimizer = optim.SGD(transformer.parameters(), lr=lr, momentum=momentum, weight_decay=l2_decay, nesterov=nesterov)

    classifier_criterion = nn.CrossEntropyLoss().cuda()
    #losses = [classifier_criterion]
    discriminator_criterion = nn.CrossEntropyLoss().cuda()
    losses = [classifier_criterion,discriminator_criterion]

    best_acc = 0
    for epoch in range(args.epochs):
        _print('### epoch : {} ###'.format(epoch))
        #train_dann(args, encoder, classifier, discriminator, src_train_loader,tgt_train_loader, optimizer, losses, epoch)
        train_transformer(args, transformer, src_train_loader, tgt_train_loader, optimizer, losses,
                    epoch,_print,writer)

        best_acc = utils.evaluate(transformer, tgt_test_loader, epoch,best_acc,_print,writer)
        #utils.save_net(args, models_td, 'tdm')
        writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
