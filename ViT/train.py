import torch
import numpy as np
import utils
import torch.optim as optim
import torch.nn as nn
import test
#from utils import save_model
#from utils import visualize
from utils import set_model_mode
import time
#import params

def train_transformer(args,transformer, source_train_loader, target_train_loader, optimizer, loss ,epoch, _print, writer):
    start = time.time()

    classifier_criterion = loss[0]
    discriminator_criterion = loss[1]

    transformer.train()

    len_dataloader = min(len(source_train_loader), len(target_train_loader))

    # start_steps = epoch * len(source_train_loader)
    # total_steps = args.epochs * len(target_train_loader)
    start_steps = epoch * len_dataloader
    total_steps = args.epochs * len_dataloader

    total_class_loss = 0
    total_domain_loss = 0

    class_correct = 0
    class_total = 0
    domain_correct = 0
    domain_total = 0
    for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
        # 길이 짧은거 만큼 돌아감, 즉 길이가 긴건 매 에폭 마다 suffle 되서 앞에 일부만 들어감

        source_image, source_label = source_data
        target_image, _ = target_data

        source_image, source_label = source_image.cuda(), source_label.cuda()
        target_image  = target_image.cuda()

        domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor).cuda() # Source : 0
        domain_target_labels = torch.ones(source_label.shape[0]).type(torch.LongTensor).cuda() # Target :1

        #p = float(batch_idx + start_steps) / total_steps
        #alpha = 2. / (1. + np.exp(-10 * p)) - 1
        #optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
        optimizer.zero_grad()

        source_class_pred, source_domain_pred = transformer(source_image)
        target_class_pred, target_domain_pred = transformer(target_image)

        class_loss = classifier_criterion(source_class_pred, source_label)
        source_domain_loss = discriminator_criterion(source_domain_pred, domain_source_labels)
        target_domain_loss = discriminator_criterion(target_domain_pred, domain_target_labels)

        total_loss = class_loss + source_domain_loss + target_domain_loss

        total_loss.backward()
        optimizer.step()

        total_class_loss += class_loss.item()
        total_domain_loss += source_domain_loss.item() + target_domain_loss.item()

        ### train accuracy ###
        pred = source_class_pred.argmax(dim=1, keepdim=True)
        class_correct += pred.eq(source_label.long().view_as(pred)).sum().item()
        class_total += source_label.size(0)

        domain_pred = source_domain_pred.argmax(dim=1, keepdim=True)
        domain_correct += domain_pred.eq(domain_source_labels.long().view_as(domain_pred)).sum().item()
        domain_pred = target_domain_pred.argmax(dim=1, keepdim=True)
        domain_correct += domain_pred.eq(domain_target_labels.long().view_as(domain_pred)).sum().item()
        domain_total += source_label.size(0)*2

    _print("(train) class loss : {:.4f} domain loss : {:.4f} train time: {:.2f}s".format(
        total_class_loss / len_dataloader,total_domain_loss / len_dataloader, time.time() - start)
    )
    _print("(train) class accuracy: {:.2f}%  domain accuracy: {:.2f}%".format((class_correct / class_total) * 100,(domain_correct / domain_total) * 100))

    writer.add_scalar('train/acc/class', (class_correct / class_total) * 100,epoch)
    writer.add_scalar('train/acc/domain', (domain_correct / domain_total) * 100, epoch)
    writer.add_scalar('train/loss/class', total_class_loss / len_dataloader, epoch)
    writer.add_scalar('train/loss/domain', total_domain_loss / len_dataloader, epoch)

    #save_model(encoder, classifier, discriminator, 'source', save_name)
    #visualize(encoder, 'source', save_name)

def train_ViT(args,transformer, source_train_loader, target_train_loader, optimizer, loss ,epoch, _print):
    start = time.time()

    classifier_criterion = loss[0]
    #discriminator_criterion = loss[1]

    transformer.train()

    len_dataloader = min(len(source_train_loader), len(target_train_loader))

    # start_steps = epoch * len(source_train_loader)
    # total_steps = args.epochs * len(target_train_loader)
    start_steps = epoch * len_dataloader
    total_steps = args.epochs * len_dataloader

    total_class_loss = 0
    total_domain_loss = 0
    for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
        # 길이 짧은거 만큼 돌아감, 즉 길이가 긴건 매 에폭 마다 suffle 되서 앞에 일부만 들어감

        source_image, source_label = source_data
        target_image, _ = target_data

        #p = float(batch_idx + start_steps) / total_steps
        #alpha = 2. / (1. + np.exp(-10 * p)) - 1

        source_image, source_label = source_image.cuda(), source_label.cuda()
        target_image = target_image.cuda()

        #optimizer = utils.optimizer_scheduler(optimizer=optimizer, p=p)
        optimizer.zero_grad()

        source_class_pred = transformer(source_image)
        class_loss = classifier_criterion(source_class_pred, source_label)
        total_loss = class_loss

        #total_loss = class_loss + source_domain_loss + target_domain_loss

        total_loss.backward()
        optimizer.step()

        total_class_loss += class_loss.item()
        #total_domain_loss += source_domain_loss.item() +target_domain_loss.item()

    # print("total_domain_loss :",total_class_loss/len_dataloader,end=" ")
    # print("total_domain_loss :",total_domain_loss/len_dataloader)
    #print("Train time: {:.2f}s".format(time.time() - start))
    _print("(train) class loss : {:.4f} domain loss : {:.4f} train time: {:.2f}s".format(
        total_class_loss / len_dataloader,0, time.time() - start)
    )
    # _print("(train) class loss : {:.4f} domain loss : {:.4f} train time: {:.2f}s".format(
    #     total_class_loss / len_dataloader,total_domain_loss / len_dataloader, time.time() - start)
    # )
    #_print("Train time: {:.2f}s".format(time.time() - start))

    #save_model(encoder, classifier, discriminator, 'source', save_name)
    #visualize(encoder, 'source', save_name)