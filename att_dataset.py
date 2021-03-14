# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import scipy.io
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import *
from reid.utils.data.sampler import RandomIdentitySampler, RandomIdentitySamplerForAtt # RandomIdentitySampler_alignedreid



def import_Market1501(dataset_dir):
    market1501_dir = os.path.join(dataset_dir,'Market-1501-v15.09.15')
    if not os.path.exists(market1501_dir):
        print('Please Download Market1501 Dataset')
    data_group = ['train','query','gallery']
    for group in data_group:
        if group == 'train':
            name_dir = os.path.join(market1501_dir , 'bounding_box_train')
        elif group == 'query':
            name_dir = os.path.join(market1501_dir, 'query')
        else:
            name_dir = os.path.join(market1501_dir, 'bounding_box_test')
        file_list=os.listdir(name_dir)
        globals()[group]={}
        for name in file_list:                       # 0002_c1s1_000451_03.jpg
            if name[-3:]=='jpg':
                id = name.split('_')[0]              # id = 0002
                if id not in globals()[group]:
                    globals()[group][id]=[]          # 创建一个含有6个(market有6个相机) 键-值 的字典
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                cam_n = int(name.split('_')[1][1])-1
                globals()[group][id][cam_n].append(os.path.join(name_dir,name))  #将图片的完整路径放入字典{'id':[[],[],[],[],[],[]]}
    return train,query,gallery


def import_Market1501Attribute(dataset_dir):
    dataset_name = 'Market-1501-v15.09.15/attribute'
    train, query, test = import_Market1501(dataset_dir)    # 返回三个字典， {'id':[[],[],[],[],[],[]]} 6个[],[]里放的是完整路径

    if not os.path.exists(os.path.join(dataset_dir, dataset_name)):
        print('Please Download the Market1501Attribute Dataset')
    train_label = ['age',
                   'backpack',
                   'bag',
                   'handbag',
                   'downblack',
                   'downblue',
                   'downbrown',
                   'downgray',
                   'downgreen',
                   'downpink',
                   'downpurple',
                   'downwhite',
                   'downyellow',
                   'upblack',
                   'upblue',
                   'upgreen',
                   'upgray',
                   'uppurple',
                   'upred',
                   'upwhite',
                   'upyellow',
                   'clothes',
                   'down',
                   'up',
                   'hair',
                   'hat',
                   'gender']

    test_label = ['age',
                  'backpack',
                  'bag',
                  'handbag',
                  'clothes',
                  'down',
                  'up',
                  'hair',
                  'hat',
                  'gender',
                  'upblack',
                  'upwhite',
                  'upred',
                  'uppurple',
                  'upyellow',
                  'upgray',
                  'upblue',
                  'upgreen',
                  'downblack',
                  'downwhite',
                  'downpink',
                  'downpurple',
                  'downyellow',
                  'downgray',
                  'downblue',
                  'downgreen',
                  'downbrown'
                  ]

    train_person_id = []
    for personid in train:  # 等价于 train.keys()
        # print(personid)
        train_person_id.append(personid)
    train_person_id.sort(key=int)   # 返回排序的仅包含id的列表:['0002', '0007', '0010']

    test_person_id = []
    for personid in test:
        test_person_id.append(personid)
    test_person_id.sort(key=int)
    test_person_id.remove('-1')
    test_person_id.remove('0000')   # 返回仅包含id的列表:['0001', '0003', '0004']  删除了id为 -1 和0000的

    f = scipy.io.loadmat(os.path.join(dataset_dir, dataset_name, 'market_attribute.mat'))

    test_attribute = {}
    train_attribute = {}
    for test_train in range(len(f['market_attribute'][0][0])):   # len=2    等于0表示test , 等于1表示train
        if test_train == 0:
            id_list_name = 'test_person_id'
            group_name = 'test_attribute'
        else:
            id_list_name = 'train_person_id'
            group_name = 'train_attribute'
        for attribute_id in range(len(f['market_attribute'][0][0][test_train][0][0])):    # len=28  27个属性和image_index(表示身份ID)
            if isinstance(f['market_attribute'][0][0][test_train][0][0][attribute_id][0][0], np.ndarray):  # 多维数组
                continue
            for person_id in range(len(f['market_attribute'][0][0][test_train][0][0][attribute_id][0])):   # len: test:750 train:751
                id = locals()[id_list_name][person_id]     #得到 id = 0001  0003  0004...
                if id not in locals()[group_name]:
                    locals()[group_name][id] = []
                locals()[group_name][id].append(
                    f['market_attribute'][0][0][test_train][0][0][attribute_id][0][person_id])     #将每个属性作为value 放到 key为id的字典中
    # 最后得到的[group_name]是字典：key:value = id标签 :27个属性标签
    # print(test_attribute)
    # print(train_attribute)

    # 将train_attribute标签与test_attribute标签顺序对应
    unified_train_atr = {}
    for k, v in train_attribute.items():
        temp_atr = [0] * len(test_label)
        for i in range(len(test_label)):
            temp_atr[i] = v[train_label.index(test_label[i])]
        unified_train_atr[k] = temp_atr
    #print(unified_train_atr) #属性标签顺序与test_attribute一致
    #print(test_attribute)
    #print(test_label)   #测试集的27个属性标签组成的列表
    return unified_train_atr, test_attribute, test_label


def import_Market1501Attribute_binary(K, logs_dir, dataset_dir):
    train_market_attr, test_market_attr, label = import_Market1501Attribute(dataset_dir)
    # 原age标签有四种值 0 1 2 3  现在改为四个单独的标签 'young', 'teenager', 'adult', 'old'
    label.pop(0)
    label.insert(0, 'young')
    label.insert(1, 'teenager')
    label.insert(2, 'adult')
    label.insert(3, 'old')
    if K == 0:
        for id in train_market_attr:
            train_market_attr[id][:] = [x - 1 for x in train_market_attr[id]]  # #将属性标签 1-2 改为 0-1
            # 原age标签有四种值 0 1 2 3  现在改为四个单独的标签 'young', 'teenager', 'adult', 'old'
            if train_market_attr[id][0] == 0:
                train_market_attr[id].pop(0)
                train_market_attr[id].insert(0, 1)
                train_market_attr[id].insert(1, 0)
                train_market_attr[id].insert(2, 0)
                train_market_attr[id].insert(3, 0)
            elif train_market_attr[id][0] == 1:
                train_market_attr[id].pop(0)
                train_market_attr[id].insert(0, 0)
                train_market_attr[id].insert(1, 1)
                train_market_attr[id].insert(2, 0)
                train_market_attr[id].insert(3, 0)
            elif train_market_attr[id][0] == 2:
                train_market_attr[id].pop(0)
                train_market_attr[id].insert(0, 0)
                train_market_attr[id].insert(1, 0)
                train_market_attr[id].insert(2, 1)
                train_market_attr[id].insert(3, 0)
            elif train_market_attr[id][0] == 3:
                train_market_attr[id].pop(0)
                train_market_attr[id].insert(0, 0)
                train_market_attr[id].insert(1, 0)
                train_market_attr[id].insert(2, 0)
                train_market_attr[id].insert(3, 1)

        for id in test_market_attr:
            test_market_attr[id][:] = [x - 1 for x in test_market_attr[id]]
            if test_market_attr[id][0] == 0:
                test_market_attr[id].pop(0)
                test_market_attr[id].insert(0, 1)
                test_market_attr[id].insert(1, 0)
                test_market_attr[id].insert(2, 0)
                test_market_attr[id].insert(3, 0)
            elif test_market_attr[id][0] == 1:
                test_market_attr[id].pop(0)
                test_market_attr[id].insert(0, 0)
                test_market_attr[id].insert(1, 1)
                test_market_attr[id].insert(2, 0)
                test_market_attr[id].insert(3, 0)
            elif test_market_attr[id][0] == 2:
                test_market_attr[id].pop(0)
                test_market_attr[id].insert(0, 0)
                test_market_attr[id].insert(1, 0)
                test_market_attr[id].insert(2, 1)
                test_market_attr[id].insert(3, 0)
            elif test_market_attr[id][0] == 3:
                test_market_attr[id].pop(0)
                test_market_attr[id].insert(0, 0)
                test_market_attr[id].insert(1, 0)
                test_market_attr[id].insert(2, 0)
                test_market_attr[id].insert(3, 1)

        #print(train_market_attr)
        #print(test_market_attr)
        #print(label)
        #返回训练集的属性标签{[id:30个属性label值]} ; 返回训练集的属性标签{[id:30个属性label值]} ; 属性标签列表仅有属性标签名字.
        #{'0002': [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], '0007': [0, 1, 0, .........
        #{'0001': [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], '0003': [0, 1, 0,..........
        #['young', 'teenager', 'adult', 'old', 'backpack', 'bag', 'handbag', 'clothes', 'down', 'up', 'hair', 'hat', 'gender', ..........
        return train_market_attr, test_market_attr, label
    else:
        ofn = os.path.join(logs_dir, "log_cluster_{}".format(K - 1), "cluster_file.txt")
        with open(ofn, 'r') as f:
            pred = f.readlines()
            train_market_attr = {}
        for pred_ in pred:
            image_path, pid_label, att_label = pred_.split("=")[0], pred_.split("=")[1], pred_.split("=")[2]  # here use att_label
            # 属性标签格式转换
            att_label_trans = []
            for att_label_ in att_label:
                # if att_label_ != '[' and att_label_ != ']' and att_label_ != ',' and att_label_ != ' ':
                if att_label_.isdigit():
                    att_label_trans.append(int(att_label_))
            # 预测的pid相同的图像的att标签不一定相同，加权
            if pid_label != '-1':
                if pid_label not in train_market_attr:
                    train_market_attr[pid_label] = []
                    train_market_attr[pid_label].append(att_label_trans)
                else:
                    train_market_attr[pid_label].append(att_label_trans)
        train_market_attr_new = {}
        for pid_key in train_market_attr.keys():
            import numpy as np
            for index, att_value in enumerate(train_market_attr[pid_key]):
                # list转np实现逐元素相加
                if index == 0:
                    att_value_temp = np.array(att_value, dtype='float')
                else:
                    att_value_temp += np.array(att_value, dtype='float')
            att_value_temp /= (index + 1)  # 取平均
            att_value_temp = np.floor(att_value_temp + 0.5)  # 四舍五入取整，np.round不行，详见https://blog.csdn.net/weixin_41712499/article/details/85208928的评论
            att_value_temp = list(att_value_temp.astype(np.int))  # 浮点转int并转list
            train_market_attr_new[pid_key] = att_value_temp
        return train_market_attr_new, label



def import_DukeMTMC(dataset_dir):
    dukemtmc_dir = os.path.join(dataset_dir, 'DukeMTMC-reID')
    if not os.path.exists(dukemtmc_dir):
        print('Please Download the DukMTMC Dataset')
    data_group = ['train','query','gallery']
    for group in data_group:
        if group == 'train':
            name_dir = os.path.join(dukemtmc_dir , 'bounding_box_train')
        elif group == 'query':
            name_dir = os.path.join(dukemtmc_dir, 'query')
        else:
            name_dir = os.path.join(dukemtmc_dir, 'bounding_box_test')
        file_list=os.listdir(name_dir)
        globals()[group]={}
        for name in file_list:
            if name[-3:]=='jpg':
                id = name.split('_')[0]
                if id not in globals()[group]:
                    globals()[group][id]=[]
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                    globals()[group][id].append([])
                cam_n = int(name.split('_')[1][1])-1
                globals()[group][id][cam_n].append(os.path.join(name_dir,name))
    return train, query, gallery


def import_DukeMTMCAttribute(dataset_dir):
    dataset_name = 'DukeMTMC-reID/attribute'
    train, query, test = import_DukeMTMC(dataset_dir)
    if not os.path.exists(os.path.join(dataset_dir, dataset_name)):
        print('Please Download the DukeMTMCATTributes Dataset')
    train_label = ['backpack',
                   'bag',
                   'handbag',
                   'boots',
                   'gender',
                   'hat',
                   'shoes',
                   'top',
                   'downblack',
                   'downwhite',
                   'downred',
                   'downgray',
                   'downblue',
                   'downgreen',
                   'downbrown',
                   'upblack',
                   'upwhite',
                   'upred',
                   'uppurple',
                   'upgray',
                   'upblue',
                   'upgreen',
                   'upbrown']

    test_label = ['boots',
                  'shoes',
                  'top',
                  'gender',
                  'hat',
                  'backpack',
                  'bag',
                  'handbag',
                  'downblack',
                  'downwhite',
                  'downred',
                  'downgray',
                  'downblue',
                  'downgreen',
                  'downbrown',
                  'upblack',
                  'upwhite',
                  'upred',
                  'upgray',
                  'upblue',
                  'upgreen',
                  'uppurple',
                  'upbrown']

    train_person_id = []
    for personid in train:
        train_person_id.append(personid)
    train_person_id.sort(key=int)

    test_person_id = []
    for personid in test:
        test_person_id.append(personid)
    test_person_id.sort(key=int)

    f = scipy.io.loadmat(os.path.join(dataset_dir, dataset_name, 'duke_attribute.mat'))

    test_attribute = {}
    train_attribute = {}
    for test_train in range(len(f['duke_attribute'][0][0])):
        if test_train == 1:
            id_list_name = 'test_person_id'
            group_name = 'test_attribute'
        else:
            id_list_name = 'train_person_id'
            group_name = 'train_attribute'
        for attribute_id in range(len(f['duke_attribute'][0][0][test_train][0][0])):
            if isinstance(f['duke_attribute'][0][0][test_train][0][0][attribute_id][0][0], np.ndarray):
                continue
            for person_id in range(len(f['duke_attribute'][0][0][test_train][0][0][attribute_id][0])):
                id = locals()[id_list_name][person_id]
                if id not in locals()[group_name]:
                    locals()[group_name][id] = []
                locals()[group_name][id].append(f['duke_attribute'][0][0][test_train][0][0][attribute_id][0][person_id])

    for i in range(8):
        train_label.insert(8, train_label[-1])
        train_label.pop(-1)

    unified_train_atr = {}
    for k, v in train_attribute.items():
        temp_atr = list(v)
        for i in range(8):
            temp_atr.insert(8, temp_atr[-1])
            temp_atr.pop(-1)
        unified_train_atr[k] = temp_atr

    unified_test_atr = {}
    for k, v in test_attribute.items():
        temp_atr = [0] * len(train_label)
        for i in range(len(train_label)):
            temp_atr[i] = v[test_label.index(train_label[i])]
        unified_test_atr[k] = temp_atr
    # two zero appear in train '0370' '0679'
    # zero_check=[]
    # for id in train_attribute:
    #    if 0 in train_attribute[id]:
    #        zero_check.append(id)
    # for i in range(len(zero_check)):
    #    train_attribute[zero_check[i]] = [1 if x==0 else x for x in train_attribute[zero_check[i]]]
    unified_train_atr['0370'][7] = 1
    unified_train_atr['0679'][7] = 2
    return unified_train_atr, unified_test_atr, train_label


def import_DukeMTMCAttribute_binary(K, logs_dir, dataset_dir):
    train_duke_attr, test_duke_attr, label = import_DukeMTMCAttribute(dataset_dir)
    # print(545, train_duke_attr.keys())
    # print(454, test_duke_attr)
    if K == 0:
        for id in train_duke_attr:
            train_duke_attr[id][:] = [x - 1 for x in train_duke_attr[id]]
        for id in test_duke_attr:
            test_duke_attr[id][:] = [x - 1 for x in test_duke_attr[id]]
        # print(545, train_duke_attr)
        # print(454, test_duke_attr)
        return train_duke_attr, test_duke_attr, label
    else:
        ofn = os.path.join(logs_dir, "log_cluster_{}".format(K - 1), "cluster_file.txt")
        with open(ofn, 'r') as f:
            pred = f.readlines()
            train_duke_attr = {}
        for pred_ in pred:
            image_path, pid_label, att_label = pred_.split("=")[0], pred_.split("=")[1], pred_.split("=")[2]  # here use att_label
            # 属性标签格式转换
            att_label_trans = []
            for att_label_ in att_label:
                # if att_label_ != '[' and att_label_ != ']' and att_label_ != ',':
                if att_label_.isdigit():
                    att_label_trans.append(int(att_label_))
            # 预测的pid相同的图像的att标签不一定相同，加权
            if pid_label != '-1':
                if pid_label not in train_duke_attr:
                    train_duke_attr[pid_label] = []
                    train_duke_attr[pid_label].append(att_label_trans)
                else:
                    train_duke_attr[pid_label].append(att_label_trans)
        train_duke_attr_new = {}
        for pid_key in train_duke_attr.keys():
            import numpy as np
            for index, att_value in enumerate(train_duke_attr[pid_key]):
                # list转np实现逐元素相加
                if index == 0:
                    att_value_temp = np.array(att_value, dtype='float')
                else:
                    att_value_temp += np.array(att_value, dtype='float')
            att_value_temp /= (index + 1)  # 取平均
            att_value_temp = np.floor(
                att_value_temp + 0.5)  # 四舍五入取整，np.round不行，详见https://blog.csdn.net/weixin_41712499/article/details/85208928的评论
            att_value_temp = list(att_value_temp.astype(np.int))  # 浮点转int并转list
            train_duke_attr_new[pid_key] = att_value_temp
        return train_duke_attr_new, label



__image_datasets = {
    'Market-1501': 'Market-1501-v15.09.15',
    'DukeMTMC-reID': 'DukeMTMC-reID',
}


def import_MarketDuke_nodistractors(K, logs_dir, data_dir, dataset_name,target_dataset_name):
    dataset_dir = os.path.join(data_dir, __image_datasets[dataset_name])

    if not os.path.exists(dataset_dir):
        print('Please Download ' + dataset_name + ' Dataset')

    dataset_dir = os.path.join(data_dir, __image_datasets[dataset_name])
    target_dataset_dir = os.path.join(data_dir, __image_datasets[target_dataset_name])
    name_dir = os.path.join(target_dataset_dir, 'bounding_box_train')
    file_list = sorted(os.listdir(name_dir))  # os.listdir()返回指定路径下的文件和文件夹列表
    target_train = {}  # globals()返回全局变量组成的字典
    target_train['data'] = []
    target_train['ids'] = []
    for name in file_list:  # 0002_c1s1_000451_03.jpg
        if name[-3:] == 'jpg':
            id = name.split('_')[0]  # 行人id = 0002
            cam = int(name.split('_')[1][1])  # cam = 1
            images = os.path.join(name_dir,name)
            if (id != '0000' and id != '-1'):
                if id not in target_train['ids']:
                    target_train['ids'].append(id)  # 把id 添加到[group]['ids']字典中
                target_train['data'].append([images, target_train['ids'].index(id), id, cam, name.split('.')[0]])

    data_group = ['train', 'query', 'gallery']
    if K == 0:
        for group in data_group:
            if group == 'train':
                name_dir = os.path.join(dataset_dir, 'bounding_box_train')
            elif group == 'query':
                name_dir = os.path.join(dataset_dir, 'query')
            else:
                name_dir = os.path.join(dataset_dir, 'bounding_box_test')
            file_list = sorted(os.listdir(name_dir))    #os.listdir()返回指定路径下的文件和文件夹列表
            globals()[group] = {}                       #globals()返回全局变量组成的字典
            globals()[group]['data'] = []
            globals()[group]['ids'] = []
            for name in file_list:                         # 0002_c1s1_000451_03.jpg
                if name[-3:] == 'jpg':
                    id = name.split('_')[0]                # 行人id = 0002
                    cam = int(name.split('_')[1][1])       # cam = 1
                    images = os.path.join(name_dir, name)  #完整的路径 images = 'G:/att_dataset/Market-1501/bounding_box_train/0002_c1s1_000451_03.jpg'
                    if (id != '0000' and id != '-1'):
                        if id not in globals()[group]['ids']:
                            globals()[group]['ids'].append(id)   # 把id 添加到[group]['ids']字典中
                        globals()[group]['data'].append(
                            [images, globals()[group]['ids'].index(id), id, cam, name.split('.')[0]])   # name= 0002_c1s1_000451_03
                        # globals()[group]['ids'].index(id)  返回id对应的索引，例如 id有四个0002，0001，0003，0002 则index = 0,1,2,0  算是表示行人的类
        return train, query, gallery, target_train
        # trian=          {'data':[['G:/att_dataset\\Market-1501\\bounding_box_train\\0002_c1s1_000451_03.jpg', 0, '0002', 1, '0002_c1s1_000451_03'],...],'ids': ['0002', '0007', '0010', '0011', '0012', '0020', '0022',...]}
        # train['data'] = [['G:/att_dataset\\Market-1501\\bounding_box_train\\0002_c1s1_000451_03.jpg', 0, '0002', 1, '0002_c1s1_000451_03'],...,['G:/att_dataset\\Market-1501\\bounding_box_train\\1500_c6s3_086567_01.jpg', 750, '1500', 6, '1500_c6s3_086567_01']]
        # train['ids'] =  ['0002', '0007', '0010', '0011', '0012', '0020', '0022',...]
    else:
        ofn = os.path.join(logs_dir, "log_cluster_{}".format(K - 1), "cluster_file.txt")
        with open(ofn, 'r') as f:
            pred = f.readlines()
        target_testset = {}
        target_testset['data'] = []
        target_testset['ids'] = []
        for pred_ in pred:
            image_path, pid_label, att_label = pred_.split("=")[0], pred_.split("=")[1], pred_.split("=")[2]  # here not use att_label
            cam = 0
            if pid_label != '-1':
                if pid_label not in target_testset['ids']:
                    target_testset['ids'].append(pid_label)
                target_testset['data'].append(
                    [image_path, target_testset['ids'].index(pid_label), pid_label, cam, os.path.basename(image_path).split('.')[0]])
        return target_testset, target_train


class Train_Dataset(data.Dataset):

    def __init__(self, K, logs_dir, data_dir, dataset_name, target_dataset_name, transforms=None, train_val='train'):
        self.train_val = train_val
        self.dataset_mame = dataset_name
        self.target_dataset_name = target_dataset_name
        if K == 0:
            train, query, gallery, target_train = import_MarketDuke_nodistractors(K, logs_dir, data_dir, dataset_name, target_dataset_name)
            # trian=  {'data':[['G:/att_dataset\\Market-1501\\bounding_box_train\\0002_c1s1_000451_03.jpg', 0, '0002', 1, '0002_c1s1_000451_03'],...],'ids': ['0002', '0007', '0010', '0011', '0012', '0020', '0022',...]}

            if dataset_name == 'Market-1501':
                train_attr, test_attr, self.label = import_Market1501Attribute_binary(K, logs_dir, data_dir)
                target_train_attr, target_test_attr, self.target_label = import_DukeMTMCAttribute_binary(K, logs_dir, data_dir)
            elif dataset_name == 'DukeMTMC-reID':
                train_attr, test_attr, self.label = import_DukeMTMCAttribute_binary(K, logs_dir, data_dir)
                target_train_attr, target_test_attr, self.target_label = import_Market1501Attribute_binary(K, logs_dir, data_dir)
            else:
                print('Input should only be Market-1501 or DukeMTMC-reID')
            # train_attr  {'0002': [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], '0007': [0, 1, 0, .........
            # test_attr   {'0001': [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], '0003': [0, 1, 0,..........
            # self.label  ['young', 'teenager', 'adult', 'old', 'backpack', 'bag', 'handbag', 'clothes', 'down', 'up', 'hair', 'hat', 'gender', ..........
        else:
            train, target_train = import_MarketDuke_nodistractors(K, logs_dir, data_dir, dataset_name, target_dataset_name)
            # print(target_train)
            if dataset_name == 'Market-1501':
                # print(11111111)
                target_train_attr, target_test_attr, self.duke_label = import_DukeMTMCAttribute_binary(0, None, data_dir)
                train_attr, self.label = import_Market1501Attribute_binary(K, logs_dir, data_dir)  # here self.label is the att label of market, we need duke att label when K != 0 for cross domain
                self.target_label = self.label
                self.label = self.duke_label
                # self.target_label = self.duke_label
            elif dataset_name == 'DukeMTMC-reID':
                # print(222222222)
                target_train_attr, target_test_attr, self.market_label = import_Market1501Attribute_binary(0, None, data_dir)
                train_attr, self.label = import_DukeMTMCAttribute_binary(K, logs_dir, data_dir)
                self.target_label = self.label
                self.label = self.market_label
                # self.target_label = self.market_label
            else:
                print('Input should only be Market-1501 or DukeMTMC-reID')
            # print(777,target_train_attr)
            # print(666,target_test_attr)
            # print(len(self.label))

        self.num_ids = len(train['ids'])
        self.num_labels = len(self.label)

        # distribution:每个属性的正样本占比
        # distribution = np.zeros(self.num_labels)
        # for k, v in train_attr.items():
        #     distribution += np.array(v)
        # self.distribution = distribution / len(train_attr)
        # print(self.distribution) #长度为30的列表


        if train_val == 'train':
            self.train_data = train['data']
            self.train_ids = train['ids']
            self.train_attr = train_attr
        elif train_val == 'target':
            self.train_data = target_train['data']
            self.train_ids = target_train['ids']
            self.train_attr = target_train_attr
        elif train_val == 'query':
            self.train_data = query['data']
            self.train_ids = query['ids']
            self.train_attr = test_attr
        elif train_val == 'gallery':
            self.train_data = gallery['data']
            self.train_ids = gallery['ids']
            self.train_attr = test_attr
        else:
            print('Input should only be train or val')

        # train['data'] = [['G:/att_dataset\\Market-1501\\bounding_box_train\\0002_c1s1_000451_03.jpg', 0, '0002', 1, '0002_c1s1_000451_03'],...,['G:/att_dataset\\Market-1501\\bounding_box_train\\1500_c6s3_086567_01.jpg', 750, '1500', 6, '1500_c6s3_086567_01']]
        # train['ids'] =  ['0002', '0007', '0010', '0011', '0012', '0020', '0022',...'1500']
        # train_attr =    {'0002': [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], '0007': [0, 1, 0, .........

        self.num_ids = len(self.train_ids)
        self.load()

        if transforms is not None:
            if train_val == 'train':
                self.transforms = transforms
            else:
                self.transforms = T.Compose([
                    T.Resize(size=(256, 128), interpolation=3),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        # print(self.train_attr)
        # print(len(self.train_data))
    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        img_path = self.train_data[index][0]     # 完整图像路径: 'G:/att_dataset\\Market-1501\\bounding_box_train\\0002_c1s1_000451_03.jpg'
        i = self.train_data[index][1]            # 索引:    0
        id = self.train_data[index][2]           # id:     0002

        cam = self.train_data[index][3]          # 相机减1: 0
        label = np.asarray(self.train_attr[id])  # 30个属性标签的数组: [0 1 0 0 0 0 0 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0]
        data = Image.open(img_path)
        data = self.transforms(data)
        name = self.train_data[index][4]         # 图像名字(去掉.jpg) 0002_c1s1_000451_03

        return data, i, label, id, cam, name, img_path # 图片，relabel_身份id,属性标签，原始身份id,cam_id,name

    def __len__(self):
        return len(self.train_data)

    def num_label(self):
        return self.num_labels

    def num_id(self):
        return self.num_ids

    def labels(self):
        return self.label

    def load(self):
        print(            "datasets loaded"                  )
        print("Train Dataset:  {}".format(self.dataset_mame))
        print("  subset  | # ids | # images | att_label")
        print("  --------------------------------------")
        print(" {:}    | {:5d} | {:8d} |{:5d} "
              .format(self.train_val, self.num_ids, len(self.train_data),len(self.label)))


class Test_Dataset(data.Dataset):
    def __init__(self, K, logs_dir, data_dir, dataset_name, target_dataset_name, transforms=None, query_gallery='query' ):
        self.query_gallery = query_gallery
        self.dataset_name =dataset_name
        train, query, gallery, target_train = import_MarketDuke_nodistractors(K, logs_dir, data_dir, dataset_name, target_dataset_name)

        if dataset_name == 'Market-1501':
            self.train_attr, self.test_attr, self.label = import_Market1501Attribute_binary(K, logs_dir, data_dir)
            # 训练集的ID-level属性， 测试集的ID-level属性，属性列表（即['young', 'old', ...]）
        elif dataset_name == 'DukeMTMC-reID':
            self.train_attr, self.test_attr, self.label = import_DukeMTMCAttribute_binary(K, logs_dir, data_dir)
        else:
            print('Input should only be Market-1501 or DukeMTMC-reID')

        if query_gallery == 'query':
            self.test_data = query['data']
            self.test_ids = query['ids']
        elif query_gallery == 'gallery':
            self.test_data = gallery['data']
            self.test_ids = gallery['ids']
        elif query_gallery == 'all':   # query + gallery
            self.test_data = gallery['data'] + query['data']
            self.test_ids = gallery['ids']
        else:
            print('Input shoud only be query or gallery;')

        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = T.Compose([
                T.Resize(size=(256, 128)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.num_ids = len(self.test_ids)
        self.load()

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据
        '''
        img_path = self.test_data[index][0]
        i = self.test_data[index][1]
        id = self.test_data[index][2]
        cam = self.test_data[index][3]
        label = np.asarray(self.test_attr[id])
        data = Image.open(img_path)
        data = self.transforms(data)
        name = self.test_data[index][4]
        return data, i, label, id, cam, name, img_path

    def __len__(self):
        return len(self.test_data)

    def labels(self):
        return self.label

    def load(self):
        print("Test Dataset:   {}".format(self.dataset_name))
        print("  {:}   | {:5d} |{:8d} "
              .format(self.query_gallery,self.num_ids, len(self.test_data)))

import random, math
class RandomErasing(object):
    """Randomly erases an image patch.

    Origin: `<https://github.com/zhunzhong07/Random-Erasing>`_

    Reference:
        Zhong et al. Random Erasing Data Augmentation.

    Args:
        probability (float, optional): probability that this operation takes place.
            Default is 0.5.
        sl (float, optional): min erasing area.
        sh (float, optional): max erasing area.
        r1 (float, optional): min aspect ratio.
        mean (list, optional): erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

def get_data(K, logs_dir, data_dir, dataset_train, dataset_test, batch_size, num_instances, workers):
    # if K != 0, load reid and att label from `logs_dir/log_cluster_(K-1)/cluster_file.txt`
    train_transformer_list = [
        transforms.Resize((256, 128), interpolation=3),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465])]
    train_transformer = transforms.Compose(train_transformer_list)

    test_transformer = transforms.Compose([
        transforms.Resize((256, 128), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    # K = 0, we from bounding_box_train get dataset and pid, att label; else get dataset and pid, att label from cluster file .txt


    if K == 0:
        # print("K==0")
        ImageFolder_train = Train_Dataset(K, logs_dir, data_dir, dataset_train, dataset_test, transforms=train_transformer, train_val='train')
        Target_ImageFolder_train = Train_Dataset(K, logs_dir, data_dir, dataset_train, dataset_test,
                                             transforms=train_transformer,
                                             train_val='target')
        # print(len(Target_ImageFolder_train))
    else:
        # print("K==",K)
        ImageFolder_train = Train_Dataset(K, logs_dir, data_dir, dataset_test, dataset_test, transforms=train_transformer, train_val='train')
        Target_ImageFolder_train = Train_Dataset(K, logs_dir, data_dir, dataset_train, dataset_test,
                                                 transforms=train_transformer,
                                                 train_val='target')
        # Target_ImageFolder_train = ImageFolder_train
        # print(444, len(Target_ImageFolder_train))
        # print(777, len(ImageFolder_train))
    # only train dataset is modified, test dataset need not, thus set K=0, and logs_dir=None
    ImageFolder_query = Test_Dataset(0, None, data_dir, dataset_test,dataset_test,transforms=test_transformer, query_gallery='query')
    ImageFolder_gallery = Test_Dataset(0, None, data_dir, dataset_test,dataset_test,transforms=test_transformer,query_gallery='gallery')

    num_label = ImageFolder_train.num_label()
    num_id = ImageFolder_train.num_id()
    sampler = RandomIdentitySamplerForAtt(ImageFolder_train,batch_size,num_instances)
    # sampler = RandomIdentitySampler(ImageFolder_train, num_instances=4)
    train_loader = torch.utils.data.DataLoader(
        ImageFolder_train,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True
    )
    # print(666, len(train_loader))
    target_train_loader = torch.utils.data.DataLoader(
        Target_ImageFolder_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    # print(555, len(target_train_loader))
    query_loader = torch.utils.data.DataLoader(
        ImageFolder_query,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )
    gallery_loader = torch.utils.data.DataLoader(
        ImageFolder_gallery,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )
    return train_loader, target_train_loader, query_loader, gallery_loader, num_label, num_id, \
           ImageFolder_train, Target_ImageFolder_train, ImageFolder_query,ImageFolder_gallery