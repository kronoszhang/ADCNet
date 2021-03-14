from __future__ import absolute_import
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False, bnneck=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,att_num_classes=0,stride =2):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained)
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

        if stride == 1:
            self.base.layer4[0].downsample[0].stride = (1,1)
            self.base.layer4[0].conv2.stride = (1,1)
        # use AdaptiveAvgPool2d
        self.base.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.localization = nn.Sequential(
            nn.Conv2d(2048, 4096, kernel_size=3),
            nn.BatchNorm2d(4096),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )


        self.fc_loc = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Linear(512, 2 * 3 * 2),
        )

        path_postion = [1, 0, 0, 0, 1/2, -1/2,
                        1, 0, 0, 0, 1/2, 1/2,]

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(path_postion, dtype=torch.float))
        
        self.down0 = nn.Sequential(nn.Conv2d(2048, 256, 1), nn.BatchNorm2d(256))
        self.down1 = nn.Sequential(nn.Conv2d(2048, 256, 1), nn.BatchNorm2d(256))
        self.fc0 = nn.Linear(256, num_classes) 
        self.fc1 = nn.Linear(256, num_classes) 
        self.embedding0 = nn.Linear(256, 128)
        self.embedding1 = nn.Linear(256, 128)
        self.bn0 = nn.BatchNorm1d(128)
        self.bn1 = nn.BatchNorm1d(128)
        #--------------------------Attr-------------------------------
        self.att_num_classes = att_num_classes
        for c in range(self.att_num_classes):
            num_ftrs = 2048
            num_bottleneck = 512
            # self.__setattr__('class_%d' % c,
            # nn.Sequential(nn.Linear(num_ftrs, num_bottleneck),
            #               nn.BatchNorm1d(num_bottleneck),
            #               nn.LeakyReLU(0.1),
            #               nn.Dropout(p=0.5),
            #               nn.Linear(num_bottleneck, 1)))
            self.__setattr__('embed_%d' % c,
                             nn.Sequential(nn.Linear(num_ftrs, num_bottleneck),
                                           nn.BatchNorm1d(num_bottleneck),))
            self.__setattr__('class_%d' % c,
                             nn.Sequential(nn.LeakyReLU(0.1),
                                           nn.Dropout(p=0.5),
                                           nn.Linear(num_bottleneck, 1)))
        # -----------------------------------------------------------
        if not self.cut_at_pooling:
            self.bnneck = bnneck
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features   #2048

            if self.bnneck:
                self.bottleneck = nn.BatchNorm1d(out_planes)
                self.bottleneck.bias.requires_grad_(False)
                init.constant_(self.bottleneck.weight, 1)
                init.constant_(self.bottleneck.bias, 0)

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
                init.constant_(self.feat_bn.weight, 1)
                init.constant_(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, K, output_feature=None):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            else:
                x = module(x)
        xs = self.localization(x)
        xs = F.adaptive_avg_pool2d(xs, (1,1))
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 2, 3)
        
        stripe0 = theta[:, 0, :, :]
        grid0 = F.affine_grid(stripe0, x.size())
        output0 = F.grid_sample(x, grid0)
        stripe1 = theta[:, 1, :, :]
        grid1 = F.affine_grid(stripe1, x.size())
        output1 = F.grid_sample(x, grid1)
        
        local_feat0 = F.adaptive_avg_pool2d(output0, (1,1))
        att_local_feat0 = local_feat0.view(local_feat0.size(0), -1)
        local_feat0 = self.down0(local_feat0)
        local_feat0 = local_feat0.view(local_feat0.size(0), -1)
        local_feat0 = local_feat0.renorm(2, 0, 1e-5).mul(1e5)
        local_feat1 = F.adaptive_avg_pool2d(output1, (1,1))
        att_local_feat1 = local_feat1.view(local_feat1.size(0), -1)
        local_feat1 = self.down1(local_feat1)
        local_feat1 = local_feat1.view(local_feat1.size(0), -1)
        local_feat1 = local_feat1.renorm(2, 0, 1e-5).mul(1e5)
       
        if K == 0:
            # suprivised
            logits0 = self.fc0(local_feat0)
            embedding0 = self.embedding0(local_feat0)
            logits1 = self.fc1(local_feat1)
            embedding1 = self.embedding1(local_feat1)
            embedding0, embedding1 = self.bn0(embedding0), self.bn1(embedding1)
        else:
            # PAUL unsuprivised
            ew0 = self.embedding0.weight
            eww0 = ew0.renorm(2, 0, 1e-5).mul(1e5)
            esim0 = local_feat0.mm(eww0.t())
            ew1 = self.embedding1.weight
            eww1 = ew1.renorm(2, 0, 1e-5).mul(1e5)
            esim1 = local_feat1.mm(eww1.t())
            embedding0, embedding1 = esim0, esim1
            embedding0, embedding1 = self.bn0(embedding0), self.bn1(embedding1)
            
        if self.cut_at_pooling:
            if K == 0:
                return x, logits0, logits1, local_feat0, local_feat1, embedding0, embedding1
            else:
                return x, local_feat0, local_feat1, embedding0, embedding1
                

        x = self.base.avgpool(x)
        x = x.view(x.size(0), -1)   # [bs,2048]

        # Att
        # att = []
        # for c in range(self.att_num_classes):
        # att.append(self.__getattr__('class_%d' % c)(x))

        # for c in range(self.att_num_classes):
        #     if c == 0:
        #         att = self.__getattr__('class_%d' % c)(x)
        #     else:
        #         att = torch.cat((att, self.__getattr__('class_%d' % c)(x) ), dim=1)
        for c in range(self.att_num_classes):
            if self.att_num_classes == 30: # Market
                if c in [0,1,2,3,29]:
                    input_att = x
                elif c in [4,6,16,17,18,19,20,21,22,23,24,26,27,28]:
                    input_att = att_local_feat0
                elif c in [5,7,8,9,10,11,12,13,14,15,25]:
                    input_att = att_local_feat1
            elif self.att_num_classes == 23: # Duke
                if c in [4]:
                    input_att = x
                elif c in [0,2,5,7,15,16,17,18,19,20,21,22]:
                    input_att = att_local_feat0
                elif c in [1,3,6,8,9,10,11,12,13,14]:
                    input_att = att_local_feat1
            if c == 0:
                att_embed = self.__getattr__('embed_%d' % c)(input_att)
                att = self.__getattr__('class_%d' % c)(att_embed)
            else:
                att_embed_ = self.__getattr__('embed_%d' % c)(input_att)
                att_embed = torch.cat((att_embed, att_embed_), dim=1)
                att = torch.cat((att, self.__getattr__('class_%d' % c)(att_embed_)), dim=1)

        # test
        if output_feature =='pool5':
            x = F.normalize(x)
            return x, att_embed, att, local_feat0, local_feat1

        if self.bnneck:
            x = self.bottleneck(x)
        feat_triplet = x

        if output_feature =='bnneck':
            x = F.normalize(x)
            return x, att_embed, att, local_feat0, local_feat1

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)

        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        if K == 0:
            return x, att, att_embed, feat_triplet, logits0, logits1, local_feat0, local_feat1, embedding0, embedding1
        else:
            return x, att, att_embed, feat_triplet, local_feat0, local_feat1, embedding0, embedding1

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)

def resnet34(**kwargs):
    return ResNet(34, **kwargs)

def resnet50(**kwargs):
    return ResNet(50, **kwargs)

def resnet101(**kwargs):
    return ResNet(101, **kwargs)

def resnet152(**kwargs):
    return ResNet(152, **kwargs)

