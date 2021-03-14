from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .loss import TripletLoss_XT, TripletLoss, CrossEntropyLoss_LabelSmooth, CenterLoss
from .utils.meters import AverageMeter
import pdb


class BaseTrainer(object):
    def __init__(self, model, criterion, num_classes):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.num_classes = num_classes
        self.criterion_x = torch.nn.CrossEntropyLoss().cuda()
        self.criterion_xent = CrossEntropyLoss_LabelSmooth(self.num_classes, epsilon=0.1, use_gpu=True, label_smooth=True)
        self.criterion_c = CenterLoss(self.num_classes, feat_dim=2048, use_gpu=True)
        center_lr = 0.5
        self.weight_c = 0.0005
        self.optimizer_center = torch.optim.SGD(self.criterion_c.parameters(), lr=center_lr)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self, epoch, K, data_loader, optimizer, warm=None, print_freq=1, warm_epoch=10):
        self.model.train()

        if warm is not None:
            warm_epoch, warm_up, warm_iteration = warm['warm_epoch'],  warm['warm_up'], warm['warm_iteration']
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()
        att_losses = AverageMeter()
        reid_losses = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            # if i > 3:
            #     break
            data_time.update(time.time() - end)

            inputs, targets, att_label = self._parse_data(inputs)
            loss, prec1, att_loss = self._forward(inputs, targets, att_label, K)
            reid_loss = loss
            # loss += 10*att_loss
            loss += 0.1 * att_loss

            losses.update(loss.item(), targets.size(0))
            reid_losses.update(reid_loss.item(), targets.size(0))
            att_losses.update(att_loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            if (epoch + 1) < warm_epoch and warm is not None:
                warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                loss *= warm_up
                warm['warm_up'] = warm_up

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for param in self.criterion_c.parameters():
            #     param.grad.data *= (1. / self.weight_c)
            # self.optimizer_center.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'ReIDLoss {:.3f} ({:.3f})\t'
                      'AttLoss {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              reid_losses.val, reid_losses.avg,
                              att_losses.val, att_losses.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

        return warm

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, pids, label, id, cam, name, img_path = inputs
        inputs = imgs.to(self.device)
        pids = pids.to(self.device)
        label = label.to(self.device)
        return inputs, pids, label.float()

    def _forward(self, inputs, targets, att_label, K):
        if K == 0: 
            outputs, att, att_feat, feat_triplet, logits0, logits1, local_feat0, local_feat1, embedding0, embedding1 = self.model(inputs, K)
        else:
            outputs, att, att_feat, feat_triplet, local_feat0, local_feat1, embedding0, embedding1 = self.model(inputs, K)
        bce_loss = torch.nn.BCELoss()
        sg = torch.nn.Sigmoid()
        att_loss = bce_loss(sg(att), att_label)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        elif isinstance(self.criterion, TripletLoss):
            # loss_t, prec = self.criterion(feat_triplet, targets)  # TripletLoss_XT
            loss_t = self.criterion(feat_triplet, targets)
            loss_t0 = self.criterion(embedding0, targets)
            loss_t1 = self.criterion(embedding1, targets)
            loss_t += loss_t0 + loss_t1
            # loss_x = self.criterion_x(outputs, targets)
            # print(torch.argmax(outputs, 1), targets)
            loss_x = self.criterion_xent(outputs, targets)
            if K == 0:
                loss_x0 = self.criterion_xent(logits0, targets)
                loss_x1 = self.criterion_xent(logits1, targets)
                loss_x += loss_x0 + loss_x1
            # print(feat_triplet.shape, targets.shape, loss_t, loss_x)
            loss_c = self.criterion_c(feat_triplet, targets)
            #loss_c0 = self.criterion_c(embedding0, targets)
            #loss_c1 = self.criterion_c(embedding1, targets)
            loss = loss_t + loss_x # + self.weight_c * loss_c
            # loss = loss_t
            prec, = accuracy(outputs.data, targets.data)
            prec = prec[0]
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec, att_loss
