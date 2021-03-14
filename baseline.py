from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch

from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid.datasets.domain_adaptation import DA
from reid import models
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from att_dataset import get_data
from reid.loss import TripletLoss
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

"""

# baseline
Mean AP: 62.7%
CMC Scores
   top-1          82.5%
   top-5          92.4%
   top-10         95.0%
   top-20         97.2%

# when add attribute
Mean AP: 64.4%
CMC Scores
   top-1          83.0%
   top-5          93.1%
   top-10         95.3%
   top-20         97.1%
   
Using this for training a baseline model with ID and attribute of pedestrian in source dataset, and load this model 
for finetuning model in target domain; If you already trained a source model with this file, you can set K=1 in
cluster-baseline.py for finetuning in target domain directly; If not, set K=0 for training model from scratch.
 
"""

def main(args):
    cudnn.benchmark = True
    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    source_train_loader, query_loader, gallery_loader, num_label, num_id, ImageFolder_train, \
    ImageFolder_query,ImageFolder_gallery = \
        get_data(args.data_dir, args.source, args.target, args.batch_size, num_instances=4, workers=8)

    # Create model
    from resnet import resnet50
    model = resnet50(pretrained=True, cut_at_pooling=False, bnneck=False, num_features=0,
                     norm=False, dropout=0, num_classes=num_id,att_num_classes=num_label, stride=2)

    # Load from checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {} "
              .format(start_epoch))
    model = nn.DataParallel(model).cuda()

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test:")
        evaluator.evaluate(query_loader, gallery_loader, ImageFolder_query, ImageFolder_gallery, args.output_feature, args.rerank)
        return
    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

    # Optimizer
    base_param_ids = set(map(id, model.module.base.parameters()))
    new_params = [p for p in model.parameters() if
                    id(p) not in base_param_ids]
    param_groups = [
        {'params': model.module.base.parameters(), 'lr_mult': 0.1},
        {'params': new_params, 'lr_mult': 1.0}]

    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Trainer
    trainer = Trainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = 40
        lr = args.lr * (0.1 ** (epoch // step_size))
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(epoch)
        trainer.train(epoch, source_train_loader, optimizer)

        save_checkpoint({
            'state_dict': model.module.state_dict(),
            'epoch': epoch + 1,
        }, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d} \n'.
              format(epoch))

    # Final test
    print('Test with best model:')
    evaluator = Evaluator(model)
    evaluator.evaluate(query_loader, gallery_loader, ImageFolder_query, ImageFolder_gallery, args.output_feature, args.rerank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="baseline")
    # source
    parser.add_argument('-s', '--source', type=str, default='Market-1501',
                        choices=['Market-1501', 'DukeMTMC-reID'])
    # target
    parser.add_argument('-t', '--target', type=str, default='Market-1501',
                        choices=['Market-1501', 'DukeMTMC-reID'])
    # images
    parser.add_argument('-b', '--batch-size', type=int, default=128, help="batch size for source") # 128
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean')
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=r"D:\reid\ReidDatasets")
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--output_feature', type=str, default='pool5')
    #random erasing
    parser.add_argument('--re', type=float, default=0)
    #  perform re-ranking
    parser.add_argument('--rerank', action='store_true', help="perform re-ranking")

    main(parser.parse_args())
