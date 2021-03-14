from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import pdb

import torch
import numpy as np

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter

from torch.autograd import Variable
from .utils import to_torch
from .utils import to_numpy
import pdb
from .utils.osutils import mkdir_if_missing


def extract_cnn_feature(model, inputs, K, output_feature=None):
    model.eval()
    inputs = to_torch(inputs)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs, att_embed, att_class, local_feat0, local_feat1 = model(inputs, K, output_feature)
        outputs = outputs.data.cpu()
        att_embed = att_embed.data.cpu()
        att_class = att_class.data.cpu()
        local_feat0, local_feat1 = local_feat0.data.cpu(), local_feat1.data.cpu()
    return outputs, att_embed, att_class, local_feat0, local_feat1


def extract_features(model, data_loader, K, print_freq=1, output_feature=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    att_features = OrderedDict()
    att_class_feature = OrderedDict()
    labels = OrderedDict()
    local_feat0_dict, local_feat1_dict = OrderedDict(), OrderedDict()

    end = time.time()
    for i, (imgs, pids, label, id, cam, fnames, img_paths) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs, att_embeds, att_classes, local_feat0, local_feat1 = extract_cnn_feature(model, imgs, K, output_feature)
        for fname, img_path, output, att_embed, att_class, local_feat0_, local_feat1_, pid in zip(fnames, img_paths, outputs, att_embeds, att_classes, local_feat0, local_feat1, pids):
            # features[fname] = output
            # labels[fname] = pid
            features[img_path] = output
            local_feat0_dict[img_path] = local_feat0_
            local_feat1_dict[img_path] = local_feat1_
            att_features[img_path] = att_embed
            att_class_feature[img_path] = att_class
            labels[img_path] = pid
            

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))
    return features, att_features, att_class_feature, labels, local_feat0_dict, local_feat1_dict


def pairwise_distance(query_features, gallery_features, query=None, gallery=None):
    x = torch.cat([query_features[path].unsqueeze(0) for data, i, label, id, cam, f, path in query], 0)
    y = torch.cat([gallery_features[path].unsqueeze(0) for data, i, label, id, cam, f, path in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20)):
    if query is not None and gallery is not None:
        query_ids = [id for data, pid, label, id, cam, name, path in query]  
        gallery_ids = [id for data, pid, label, id, cam, name, path in gallery]
        query_cams = [cam for data, pid, label, id, cam, name, path in query]
        gallery_cams = [cam for data, pid, label, id, cam, name, path in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)
    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute CMC scores
    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'
              .format(k, cmc_scores['market1501'][k - 1]))

    return cmc_scores['market1501'][0]


def reranking(query_features, gallery_features, query=None, gallery=None, k1=20, k2=6, lamda_value=0.3):
        x = torch.cat([query_features[f].unsqueeze(0) for f, _, _ in query], 0)
        y = torch.cat([gallery_features[f].unsqueeze(0) for f, _, _ in gallery], 0)
        feat = torch.cat((x, y))
        query_num, all_num = x.size(0), feat.size(0)
        feat = feat.view(all_num, -1)

        dist = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num)
        dist = dist + dist.t()
        dist.addmm_(1, -2, feat, feat.t())

        original_dist = dist.numpy()
        all_num = original_dist.shape[0]
        original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
        V = np.zeros_like(original_dist).astype(np.float16)
        initial_rank = np.argsort(original_dist).astype(np.int32)

        print('starting re_ranking')
        for i in range(all_num):
            # k-reciprocal neighbors
            forward_k_neigh_index = initial_rank[i, :k1 + 1]
            backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
            fi = np.where(backward_k_neigh_index == i)[0]
            k_reciprocal_index = forward_k_neigh_index[fi]
            k_reciprocal_expansion_index = k_reciprocal_index
            for j in range(len(k_reciprocal_index)):
                candidate = k_reciprocal_index[j]
                candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
                candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                   :int(np.around(k1 / 2)) + 1]
                fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
                candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
                if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                        candidate_k_reciprocal_index):
                    k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

            k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
            weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
            V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
        original_dist = original_dist[:query_num, ]
        if k2 != 1:
            V_qe = np.zeros_like(V, dtype=np.float16)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
            V = V_qe
            del V_qe
        del initial_rank
        invIndex = []
        for i in range(all_num):
            invIndex.append(np.where(V[:, i] != 0)[0])

        jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

        for i in range(query_num):
            temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
            indNonZero = np.where(V[i, :] != 0)[0]
            indImages = []
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                                   V[indImages[j], indNonZero[j]])
            jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

        final_dist = jaccard_dist * (1 - lamda_value) + original_dist * lamda_value
        del original_dist
        del V
        del jaccard_dist
        final_dist = final_dist[:query_num, query_num:]
        return final_dist

import numpy as np
from scipy.spatial.distance import cdist

def k_reciprocal_neigh( initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def re_ranking_init(query_feature, gallery_feature, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
    q_q_dist = np.dot(query_feature, np.transpose(query_feature))
    g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = 2. - 2 * original_dist   #np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    #initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition( original_dist, range(1,k1+1) )

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh( initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh( initial_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)

    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist

def re_ranking(input_feature_source, input_feature, k1=20, k2=6, lambda_value=0.2, MemorySave=False, Minibatch=2000,
               no_rerank=False):
    MemorySave = True
    all_num_source = input_feature_source.shape[0]
    # query_num = probFea.shape[0]
    all_num = input_feature.shape[0]
    # feat = np.append(probFea,galFea,axis = 0)
    feat = input_feature.numpy().astype(np.float16)

    print('computing source distance...')
    sour_tar_dist = np.power(
        cdist(input_feature, input_feature_source), 2).astype(np.float16)
    sour_tar_dist = 1 - np.exp(-sour_tar_dist)
    source_dist_vec = np.min(sour_tar_dist, axis=1)
    source_dist_vec = source_dist_vec / np.max(source_dist_vec)
    source_dist = np.zeros([all_num, all_num])
    for i in range(all_num):
        source_dist[i, :] = source_dist_vec + source_dist_vec[i]
    del sour_tar_dist
    del source_dist_vec
    del input_feature
    del input_feature_source

    print('computing original distance...')
    if MemorySave:
        original_dist = np.zeros(shape=[all_num, all_num], dtype=np.float16)
        i = 0
        while True:
            it = i + Minibatch
            if it < np.shape(feat)[0]:
                original_dist[i:it, ] = np.power(cdist(feat[i:it, ], feat), 2).astype(np.float16)
            else:
                original_dist[i:, :] = np.power(cdist(feat[i:, ], feat), 2).astype(np.float16)
                break
            i = it
    else:
        original_dist = cdist(feat, feat).astype(np.float16)
        original_dist = np.power(original_dist, 2).astype(np.float16)
    # del feat
    euclidean_dist = original_dist
    if no_rerank:
        return euclidean_dist, None
    gallery_num = original_dist.shape[0]  # gallery_num=all_num
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)  ## default axis=-1.

    print('starting re_ranking...')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,
                                :k1 + 1]  ## k1+1 because self always ranks first. forward_k_neigh_index.shape=[k1+1].  forward_k_neigh_index[0] == i.
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,
                                 :k1 + 1]  ##backward.shape = [k1+1, k1+1]. For each ele in forward_k_neigh_index, find its rank k1 neighbors
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]  ## get R(p,k) in the paper
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    # original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    """jaccard_dist = np.zeros_like(original_dist,dtype = np.float16)

    for i in range(all_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float16)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2-temp_min)"""

    def compute_jaccard_dist(target_features, k1=20, k2=6, print_flag=True,
                             lambda_value=0, source_features=None, use_gpu=False):
        import time, torch
        end = time.time()
        N = target_features.shape[0]
        # N = target_features.size(0)
        if (use_gpu):
            # accelerate matrix distance computing
            target_features = target_features.cuda()
            if (source_features is not None):
                source_features = source_features.cuda()

        if ((lambda_value > 0) and (source_features is not None)):
            M = source_features.size(0)
            sour_tar_dist = torch.pow(target_features, 2).sum(dim=1, keepdim=True).expand(N, M) + \
                            torch.pow(source_features, 2).sum(dim=1, keepdim=True).expand(M, N).t()
            sour_tar_dist.addmm_(1, -2, target_features, source_features.t())
            sour_tar_dist = 1 - torch.exp(-sour_tar_dist)
            sour_tar_dist = sour_tar_dist.cpu()
            source_dist_vec = sour_tar_dist.min(1)[0]
            del sour_tar_dist
            source_dist_vec /= source_dist_vec.max()
            source_dist = torch.zeros(N, N)
            for i in range(N):
                source_dist[i, :] = source_dist_vec + source_dist_vec[i]
            del source_dist_vec

        if print_flag:
            print('Computing original distance...')
        target_features = target_features.float()
        original_dist = torch.pow(target_features, 2).sum(dim=1, keepdim=True) * 2
        original_dist = original_dist.expand(N, N) - 2 * torch.mm(target_features, target_features.t())
        original_dist /= original_dist.max(0)[0]
        original_dist = original_dist.t()
        initial_rank = torch.argsort(original_dist, dim=-1)

        original_dist = original_dist.cpu()
        initial_rank = initial_rank.cpu()
        all_num = gallery_num = original_dist.size(0)

        del target_features
        if (source_features is not None):
            del source_features

        if print_flag:
            print('Computing Jaccard distance...')

        nn_k1 = []
        nn_k1_half = []
        for i in range(all_num):
            nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
            nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

        V = torch.zeros(all_num, all_num)
        for i in range(all_num):
            k_reciprocal_index = nn_k1[i]
            k_reciprocal_expansion_index = k_reciprocal_index
            for candidate in k_reciprocal_index:
                candidate_k_reciprocal_index = nn_k1_half[candidate]
                if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                        candidate_k_reciprocal_index)):
                    k_reciprocal_expansion_index = torch.cat(
                        (k_reciprocal_expansion_index, candidate_k_reciprocal_index))

            k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)  ## element-wise unique
            weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])
            V[i, k_reciprocal_expansion_index] = weight / torch.sum(weight)

        if k2 != 1:
            k2_rank = initial_rank[:, :k2].clone().view(-1)
            V_qe = V[k2_rank]
            V_qe = V_qe.view(initial_rank.size(0), k2, -1).sum(1)
            V_qe /= k2
            V = V_qe
            del V_qe
        del initial_rank

        invIndex = []
        for i in range(gallery_num):
            invIndex.append(torch.nonzero(V[:, i])[:, 0])  # len(invIndex)=all_num

        jaccard_dist = torch.zeros_like(original_dist)
        for i in range(all_num):
            temp_min = torch.zeros(1, gallery_num)
            indNonZero = torch.nonzero(V[i, :])[:, 0]
            indImages = []
            indImages = [invIndex[ind] for ind in indNonZero]
            for j in range(len(indNonZero)):
                temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + torch.min(V[i, indNonZero[j]],
                                                                                  V[indImages[j], indNonZero[j]])
            jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
        del invIndex

        del V

        pos_bool = (jaccard_dist < 0)
        jaccard_dist[pos_bool] = 0.0
        if print_flag:
            print("Time cost: {}".format(time.time() - end))

        if (lambda_value > 0):
            return jaccard_dist * (1 - lambda_value) + source_dist * lambda_value
        else:
            return jaccard_dist

    import torch
    from torch.autograd import Variable
    feat = Variable(torch.from_numpy(feat))
    jaccard_dist = compute_jaccard_dist(feat, use_gpu=False)
    del feat
    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    del pos_bool

    # final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    # del temp_min
    del forward_k_neigh_index
    del backward_k_neigh_index
    del fi
    del fi_candidate
    del k_reciprocal_index
    del k_reciprocal_expansion_index
    del candidate
    del candidate_forward_k_neigh_index
    del candidate_backward_k_neigh_index
    del candidate_k_reciprocal_index
    del weight
    print(type(jaccard_dist), type(lambda_value), type(source_dist))
    final_dist = jaccard_dist * (1 - lambda_value) + Variable(torch.from_numpy(source_dist)) * lambda_value
    del jaccard_dist
    # final_dist = final_dist[:query_num,query_num:]
    return euclidean_dist, final_dist.numpy()
class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, K, logs_dir, true_source_train_loader, True_ImageFolder_train, target_train_loader,Target_ImageFolder_train,query_loader, gallery_loader, query, gallery, output_feature=None, rerank=False):
        import time
        start_time = time.time()
        true_source_train_features, _, _, _, _, _ = extract_features(self.model,true_source_train_loader,K, 50,output_feature)
        target_train_features, target_train_att_features, target_train_att_class, _ , target_train_local0, target_train_local1= extract_features(self.model, target_train_loader, K, 50, output_feature)
        query_features, query_att_features, query_att_class, _, query_local0, query_local1 = extract_features(self.model, query_loader, K, 50, output_feature)
        gallery_features, gallery_att_features, gallery_att_class, _, gallery_local0, gallery_local1 = extract_features(self.model, gallery_loader, K, 50, output_feature)
        if rerank:
            distmat = reranking(query_features, gallery_features, query, gallery)
        else:
            distmat = pairwise_distance(query_features, gallery_features, query, gallery)
        end_time = time.time()
        print("Extracting feature and rerank={} finished, {:.2f} s time used ...".format(rerank, end_time-start_time))
        # cluster
        def hdbscan(feat, min_samples=10):
            import hdbscan
            db = hdbscan.HDBSCAN(min_cluster_size=min_samples)
            labels_ = db.fit_predict(feat)
            return labels_

        def hdbscan_dist(dist, min_samples=10):
            import hdbscan
            db = hdbscan.HDBSCAN(min_cluster_size=min_samples, metric='precomputed')
            labels_ = db.fit_predict(dist)
            return labels_

        def cluster_re_ranking(input_feature_source, input_feature, k1=20, k2=6, lambda_value=0.1):
            from scipy.spatial.distance import cdist
            all_num = input_feature.shape[0]
            feat = input_feature.astype(np.float16)
            if lambda_value != 0:
                print('Computing source distance...')
                all_num_source = input_feature_source.shape[0]
                sour_tar_dist = np.power(
                    cdist(input_feature, input_feature_source), 2).astype(np.float16)
                sour_tar_dist = 1 - np.exp(-sour_tar_dist)
                source_dist_vec = np.min(sour_tar_dist, axis=1)
                source_dist_vec = source_dist_vec / np.max(source_dist_vec)
                source_dist = np.zeros([all_num, all_num])
                for i in range(all_num):
                    source_dist[i, :] = source_dist_vec + source_dist_vec[i]
                del sour_tar_dist
                del source_dist_vec
            print('Computing original distance...')
            original_dist = cdist(feat, feat).astype(np.float16)
            original_dist = np.power(original_dist, 2).astype(np.float16)
            del feat
            euclidean_dist = original_dist
            gallery_num = original_dist.shape[0]  # gallery_num=all_num
            original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
            V = np.zeros_like(original_dist).astype(np.float16)
            initial_rank = np.argsort(original_dist).astype(np.int32)  ## default axis=-1.
            print('Starting re_ranking...')
            for i in range(all_num):
                # k-reciprocal neighbors
                forward_k_neigh_index = initial_rank[i,
                                        :k1 + 1]  ## k1+1 because self always ranks first. forward_k_neigh_index.shape=[k1+1].  forward_k_neigh_index[0] == i.
                backward_k_neigh_index = initial_rank[forward_k_neigh_index,
                                         :k1 + 1]  ##backward.shape = [k1+1, k1+1]. For each ele in forward_k_neigh_index, find its rank k1 neighbors
                fi = np.where(backward_k_neigh_index == i)[0]
                k_reciprocal_index = forward_k_neigh_index[fi]  ## get R(p,k) in the paper
                k_reciprocal_expansion_index = k_reciprocal_index
                for j in range(len(k_reciprocal_index)):
                    candidate = k_reciprocal_index[j]
                    candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
                    candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                                       :int(np.around(k1 / 2)) + 1]
                    fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
                    candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
                    if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                            candidate_k_reciprocal_index):
                        k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,
                                                                 candidate_k_reciprocal_index)
                k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
                weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
                V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
            # original_dist = original_dist[:query_num,]
            if k2 != 1:
                V_qe = np.zeros_like(V, dtype=np.float16)
                for i in range(all_num):
                    V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
                V = V_qe
                del V_qe
            del initial_rank
            invIndex = []
            for i in range(gallery_num):
                invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num
            jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
            for i in range(all_num):
                temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
                indNonZero = np.where(V[i, :] != 0)[0]
                indImages = []
                indImages = [invIndex[ind] for ind in indNonZero]
                for j in range(len(indNonZero)):
                    temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                                       V[indImages[j], indNonZero[j]])
                jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
            pos_bool = (jaccard_dist < 0)
            jaccard_dist[pos_bool] = 0.0
            if lambda_value == 0:
                return jaccard_dist
            else:
                final_dist = jaccard_dist * (1 - lambda_value) + source_dist * lambda_value
                return final_dist

        import os
        fpath = os.path.join(logs_dir, "log_cluster_{}".format(K), 'checkpoint.pth.tar')
        mkdir_if_missing(os.path.dirname(fpath))

        # test_f = torch.cat((qf, gf), 0)  # N * 2048
        # att_test_f = torch.cat((att_qf, att_gf), 0)
        # att_test_label = torch.argmax(att_test_f, -1)
        # att_test_f = att_test_f.view(att_test_f.size(0), -1)  # N * 60   60=30*2
        # from sklearn.decomposition import PCA
        # pca_sk = PCA(n_components=256)
        # test_f = pca_sk.fit_transform(test_f)
        # print(att_test_f.shape)
        # if K <= 5:
        #     cluster_feat = test_f
        # else:
        #     cluster_feat = torch.cat((test_f, att_test_f), 1)
        testset_features = []
        att_testset_features = []
        att_class_result = []
        testset_names = []
        for img_path, feature in query_features.items():
            testset_features.append(feature.unsqueeze(0))
            testset_names.append(img_path)
            att_testset_features.append(query_att_features[img_path].unsqueeze(0))
            att_class_result.append(query_att_class[img_path].unsqueeze(0))
        for img_path, feature in gallery_features.items():
            testset_features.append(feature.unsqueeze(0))
            testset_names.append(img_path)
            att_testset_features.append(gallery_att_features[img_path].unsqueeze(0))
            att_class_result.append(gallery_att_class[img_path].unsqueeze(0))
        testset_features = torch.cat(testset_features, 0)
        att_testset_features = torch.cat(att_testset_features, 0)
        att_class_result = torch.cat(att_class_result, 0)
        sg = torch.nn.Sigmoid()
        att_class_result = sg(att_class_result) > 0.5  
        # pred = hdbscan_dist(distmat, min_samples=5)
        print(att_testset_features.shape)  # torch.Size([19889, 15360])
        print(att_class_result.shape)  # torch.Size([19889, 30])
        # cluster_feat = testset_features  # 33.1% & 44.6%  about use 20min
        # from sklearn.decomposition import PCA
        # pca_sk = PCA(n_components=2048)
        # att_testset_features = pca_sk.fit_transform(att_testset_features)
        # att_testset_features = torch.tensor(att_testset_features).float()
        # cluster_feat = att_testset_features
        target_trainset_features = []
        target_trainset_att_features = []
        target_trainset_att_class = []
        target_trainset_names = []
        for img_path, feature in target_train_features.items():
            target_trainset_features.append(feature.unsqueeze(0))
            target_trainset_names.append(img_path)
            target_trainset_att_features.append(target_train_att_features[img_path].unsqueeze(0))
            target_trainset_att_class.append(target_train_att_class[img_path].unsqueeze(0))
        true_source_trainset_features = []
        for img_path, feature in true_source_train_features.items():
            true_source_trainset_features.append(feature.unsqueeze(0))
            
        target_trainset_local_features0 = []
        for img_path, feature in target_train_local0.items():
            target_trainset_local_features0.append(feature.unsqueeze(0))
        target_trainset_local_features1 = []
        for img_path, feature in target_train_local1.items():
            target_trainset_local_features1.append(feature.unsqueeze(0))          
            
        true_source_trainset_features = torch.cat(true_source_trainset_features, 0)
        target_trainset_features = torch.cat(target_trainset_features, 0)
        target_trainset_local_features0 = torch.cat(target_trainset_local_features0, 0)
        target_trainset_local_features1 = torch.cat(target_trainset_local_features1, 0)
        target_trainset_features = torch.cat((target_trainset_features, target_trainset_local_features0, target_trainset_local_features1), 1)
        target_trainset_att_features = torch.cat(target_trainset_att_features, 0)
        target_trainset_att_class = torch.cat(target_trainset_att_class, 0)
        sg = torch.nn.Sigmoid()
        target_trainset_att_class = sg(target_trainset_att_class) > 0.5 
        print(target_trainset_features.shape, target_trainset_att_features.shape, target_trainset_att_class.shape)
        #from sklearn.decomposition import PCA
        #pca_sk = PCA(n_components=10)
        #target_trainset_features = pca_sk.fit_transform(target_trainset_features)
        """no_rerank = False
        euclidean_dist, rerank_dist = re_ranking(
            true_source_trainset_features,
            target_trainset_features,
            lambda_value=0.1, no_rerank=no_rerank
        )
        rerank_dist_list = []
        euclidean_dist_list = []
        rerank_dist_list.append(rerank_dist)
        # euclidean_dist_list.append([])
        euclidean_dist_list.append(euclidean_dist)
        del euclidean_dist
        del rerank_dist
        cluster_list = []
        labels_list = []
        for s in range(len(rerank_dist_list)):
            if no_rerank:
                tmp_dist = euclidean_dist_list[s]
            else:
                tmp_dist = rerank_dist_list[s]
            ####HDBSCAN cluster
            pred = hdbscan_dist(tmp_dist, min_samples=5)
            # pred = hdbscan(cluster_feat, min_samples=5)
            # cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=8)"""
            
        
        if K <10:
            # from sklearn.decomposition import PCA
            # pca_sk = PCA(n_components=2048)
            # att_testset_features = pca_sk.fit_transform(att_testset_features)
            #target_trainset_att_features = torch.tensor(target_trainset_att_features).float()
            #cluster_feat = torch.cat((target_trainset_features, target_trainset_att_features), 1)
            cluster_feat = target_trainset_features
        else:
            # cluster_feat = target_trainset_features
            target_trainset_att_features = torch.tensor(target_trainset_att_features).float()
            cluster_feat = torch.cat((target_trainset_features, target_trainset_att_features), 1)
        #from sklearn.decomposition import PCA
        #pca_sk = PCA(n_components=10)
        #cluster_feat = pca_sk.fit_transform(cluster_feat)
            
        print(cluster_feat.shape)
        pred = hdbscan(cluster_feat, min_samples=4) #5


        """if K < 5:
            # from sklearn.decomposition import PCA
            # pca_sk = PCA(n_components=2048)
            # att_testset_features = pca_sk.fit_transform(att_testset_features)
            att_testset_features = torch.tensor(att_testset_features).float()
            cluster_feat = torch.cat((testset_features, att_testset_features), 1)
        else:
            cluster_feat = testset_features"""
        # if K < 2:
        #     from sklearn.decomposition import PCA
        #     pca_sk = PCA(n_components=2048)
        #     att_testset_features = pca_sk.fit_transform(att_testset_features)
        #     att_testset_features = torch.tensor(att_testset_features).float()
        #     cluster_feat = torch.cat((testset_features, att_testset_features), 1)
        # else:  
        #     cluster_feat = testset_features
        
        # post process
        valid = np.where(pred != -1)
        _, unique_idx = np.unique(pred[valid], return_index=True)
        pred_unique = pred[valid][np.sort(unique_idx)]
        pred_mapping = dict(zip(list(pred_unique), range(pred_unique.shape[0])))
        pred_mapping[-1] = -1
        pred = np.array([pred_mapping[p] for p in pred])
        print("Discard ratio: {:.4g}".format(1 - len(valid[0]) / float(len(pred))))
        # save
        reid_ofn = os.path.join(logs_dir, "log_cluster_{}".format(K), "cluster_file.txt")
        with open(reid_ofn, 'w') as f:
            f.writelines(["{}={}={}\n".format(target_trainset_names[index], l, list(target_trainset_att_class[index].numpy().astype(int))) for index, l in enumerate(pred)])
            # f.writelines(["{}={}={}\n".format(testset_names[index], l, list(att_class_result[index].numpy())) for index, l in enumerate(pred)])
        print("Save as: {}".format(reid_ofn))
        num_class_valid = len(np.unique(pred[np.where(pred != -1)]))
        pred_with_singular = pred.copy()
        # to assign -1 with new labels
        pred_with_singular[np.where(pred == -1)] = np.arange(num_class_valid, num_class_valid + (pred == -1).sum())
        print("#cluster: {}".format(len(np.unique(pred_with_singular))))
        print("Cluster time: {:.2f} s ...".format(time.time() - end_time))

        return evaluate_all(distmat, query=query, gallery=gallery)
