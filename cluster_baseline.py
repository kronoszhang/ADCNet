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
from reid.utils.serialization import load_checkpoint, save_checkpoint, load_checkpoint_except_reid_class
from att_dataset import get_data
from reid.loss import TripletLoss
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
"""
# Paper name: Attribute Driven Cluster for Cross-Domain Person Re-Identification. (ADCNet)
# Why work: ReID is open-set, thus ID feature can not generalize well in cross domain dataset, while attribute is
            close-set task, thus attribute is benefit for ReID. Furthermore, we split person to upper-body, low-body
            and whole body, thus recognize different attribute in different area for avoiding confusion, just like 
            the human, see more in Fig2.
# Our code is based-on paper: Generalizing a person retrieval model hetero-and homogeneously.(ZhongZhun,et al, CVPR2019)
            in link: https://github.com/zhunzhong07/HHL.git
"""

def main(args):
    cudnn.benchmark = True
    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    K = 0  # K = 0 for source training, others for target domain fine-tuning
    while K < 30:
        print("================================================")
        print("*****         The {}-th Finetune ...        *****".format(K))
        print("================================================")
        # Create data loaders
        source_train_loader, target_train_loader, query_loader, gallery_loader, num_label, num_id, ImageFolder_train, Target_ImageFolder_train, ImageFolder_query,ImageFolder_gallery = \
            get_data(K, args.logs_dir, args.data_dir, args.source, args.target, args.batch_size, num_instances=4, workers=8)
        print(num_id, num_label)
        # Create model
        from resnet import resnet50
        model = resnet50(pretrained=True, cut_at_pooling=False, bnneck=True, num_features=0,
                         norm=False, dropout=0, num_classes=num_id, att_num_classes=num_label, stride=1)
        if K != 0:
            checkpoint = load_checkpoint_except_reid_class(model, os.path.join(args.logs_dir, "log_cluster_{}".format(K-1), "checkpoint.pth.tar"))
            start_epoch = checkpoint['epoch']
            print("=> Start epoch {} ".format(start_epoch))
        if args.resume:
            # load the total model
            checkpoint = load_checkpoint(os.path.join(args.logs_dir, "log_cluster_{}".format(0), "checkpoint.pth.tar"))
            model.load_state_dict(checkpoint['state_dict'])
            start_epoch = checkpoint['epoch']
            print("=> Start epoch {} ".format(start_epoch))
        model = nn.DataParallel(model).cuda()

        # Criterion
        # criterion = nn.CrossEntropyLoss().cuda()  # only softmax loss
        criterion = TripletLoss(margin=0.3).cuda()  # softmax loss + triplet loss + center loss

        # Optimizer
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                        id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]

        """
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
        """
        optimizer = torch.optim.Adam(param_groups,
                                     lr=3.5e-4,
                                     weight_decay=5e-04,
                                     betas=(0.9, 0.99),
                                     )
        # Trainer
        trainer = Trainer(model, criterion, num_id)

        # Schedule learning rate
        def adjust_lr(epoch):
            step_size = 40
            lr = args.lr * (0.1 ** (epoch // step_size))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

        warm_epoch = 10
        warm_up = 0.1
        warm_iteration = len(source_train_loader) * warm_epoch  # warm up in first 10 epoch, same effect
        warm = {'warm_epoch': warm_epoch,
                'warm_up': warm_up,
                'warm_iteration': warm_iteration, }
        # Start training
        for epoch in range(args.epochs):
            adjust_lr(epoch)
            warm = trainer.train(epoch, source_train_loader, optimizer, warm=warm, warm_epoch=10, print_freq=50)

            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
            }, fpath=osp.join(args.logs_dir, "log_cluster_{}".format(K), 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d} \n'.
                  format(epoch))

        # Final test
        print('Test with best model:')
        evaluator = Evaluator(model)
        evaluator.evaluate(K, args.logs_dir, target_train_loader,Target_ImageFolder_train,  query_loader, gallery_loader, ImageFolder_query, ImageFolder_gallery, args.output_feature, args.rerank)
        K += 1


if __name__ == '__main__':
    """
    When using weak baseline (only softmax loss, 6 tricks not used) and batch_size is set to 128:
         we get mAP=22.1%, Rank@1=44.6%
    When using wead baseline, and add attribute feature(use PCA to reduce dim of attribute feature to 2048) 
         and batch_size is set to 64, then concat reid feature and attribute feature for cluster feature for DBSCAN cluster,
         we get mAP=34.8%, Rank@1=48.0%
    When using strong baseline while not using attrubute, we get mAP=25.7%, Rank@1=41.4%; And when add attribute, better 
         performance would get, we get mAP=55.3%, Rank@1=73.6%.
        
    All result is reported in Market2Duke.
    """

    parser = argparse.ArgumentParser(description="baseline")
    # source
    parser.add_argument('-s', '--source', type=str, default='Market-1501',
                        choices=['Market-1501', 'DukeMTMC-reID'])
    # target
    parser.add_argument('-t', '--target', type=str, default='DukeMTMC-reID',
                        choices=['Market-1501', 'DukeMTMC-reID'])
    # images
    parser.add_argument('-b', '--batch-size', type=int, default=64, help="batch size for source")  # 128
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
    parser.add_argument('--output_feature', type=str, default='bnneck')
    #random erasing
    parser.add_argument('--re', type=float, default=0)
    #  perform re-ranking
    parser.add_argument('--rerank', action='store_true', help="perform re-ranking")

    main(parser.parse_args())

