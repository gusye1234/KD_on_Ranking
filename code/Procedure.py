'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import os
import world
import torch
import utils
import model
import dataloader
import multiprocessing
import numpy as np
from time import time
from tqdm import tqdm
from pprint import pprint
from sample import DistillSample
from sample import LogitsSample
from model import PairWiseModel, BasicModel
from sample import UniformSample_DNS_deter, UniformSample_DNS_batch
from sample import UniformSample_original,DNS_sampling_neg, DNS_sampling_batch
from utils import time2str, timer

item_count = None

CORES = multiprocessing.cpu_count() // 2
# ----------------------------------------------------------------------------
def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    global item_count
    if item_count is None:
        item_count = torch.zeros(dataset.m_items)
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    allusers = list(range(dataset.n_users))
    S, sam_time = UniformSample_original(allusers, dataset)
    print(f"BPR[sample time][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.DEVICE)
    posItems = posItems.to(world.DEVICE)
    negItems = negItems.to(world.DEVICE)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        if world.ALLDATA:
            weights = utils.getTestweight(batch_users, batch_pos, dataset)
        else:
            weights = None
        item_count[batch_neg] += 1
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg, weights=weights)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    return f"[BPR[aver loss{aver_loss:.3e}]"
# ----------------------------------------------------------------------------
def BPR_train_DNS_neg(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel: PairWiseModel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    S,sam_time = UniformSample_DNS_deter(dataset, world.DNS_K)
    # S,sam_time = UniformSample_DNS_neg_multi(allusers, dataset, world.DNS_K)
    print(f"DNS[pre-sample][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2:]).long()
    print(negItems.shape)
    users = users.to(world.DEVICE)
    posItems = posItems.to(world.DEVICE)
    negItems = negItems.to(world.DEVICE)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    DNS_time = time()
    DNS_time1 = 0.
    DNS_time2 = 0.
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        if world.ALLDATA:
            weights = utils.getTestweight(batch_users, batch_pos, dataset)
        else:
            weights = None
        batch_neg, sam_time = DNS_sampling_neg(batch_users, batch_neg, dataset, Recmodel)
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg, weights=weights)
        DNS_time1 += sam_time[0]
        DNS_time2 += sam_time[2]
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    print(f"DNS[sampling][{time()-DNS_time:.1f}={DNS_time1:.2f}+{DNS_time2:.2f}]")
    aver_loss = aver_loss / total_batch
    return f"[BPR[aver loss{aver_loss:.3e}]"
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def Distill_train(dataset, student, sampler, loss_class, epoch, neg_k=1, w=None):
    global item_count
    sampler : DistillSample
    bpr: utils.BPRLoss = loss_class
    student.train()
    if item_count is None:
        item_count = torch.zeros(dataset.m_items)
    # S,sam_time = UniformSample_DNS_deter(allusers, dataset, world.DNS_K)
    (S,
     negItems,
     negScores, 
     negScores_teacher, 
     sam_time) = sampler.UniformSample_DNS_batch(epoch)

    print(f"[pre-sample][{sam_time[0]:.1f}={time2str(sam_time[1:])}]")
    users = S[:, 0].long()
    posItems = S[:, 1].long()
    # negItems = S[:, 2:].long()
    negItems = negItems.long()
    print(negItems.shape, negScores.shape, negScores_teacher.shape)
    (users, 
     posItems, 
     negItems, 
     negScores,
     negScores_teacher) = utils.TO(users, posItems, negItems, negScores, negScores_teacher)
    (users, 
     posItems, 
     negItems, 
     negScores,
     negScores_teacher) = utils.shuffle(users, posItems, negItems, negScores, negScores_teacher)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    DNS_time = time()
    DNS_time1 = 0.
    DNS_time2 = 0.
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg,
          batch_scores,
          batch_scores_teacher)) in enumerate(utils.minibatch(users,
                                                              posItems,
                                                              negItems,
                                                              negScores,
                                                              negScores_teacher,
                                                              batch_size=world.config['bpr_batch_size'])):
        # batch_neg, sam_time = DNS_sampling_neg(batch_users, batch_neg, dataset, Recmodel)
        # batch_neg, sam_time = DNS_sampling_batch(batch_neg, batch_scores)
        batch_neg, weights, samtime= sampler.Sample(batch_neg, batch_pos,batch_users,batch_scores, batch_scores_teacher)
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg, weights=weights)
        item_count[batch_neg] += 1
        DNS_time1 += sam_time[0]
        DNS_time2 += sam_time[2]
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    np.savetxt(os.path.join(world.CODE_PATH, f"counts/count_{world.dataset}_{world.DNS_K}.txt"),item_count.numpy())
    print(f"[sampling][{time()-DNS_time:.1f}={DNS_time1:.2f}+{DNS_time2:.2f}]")
    aver_loss = aver_loss / total_batch
    return f"BPR[aver loss{aver_loss:.3e}]"      

def Logits_DNS(dataset, student, sampler, loss_class, epoch, neg_k=1, w=None):
    sampler : LogitsSample
    bpr: utils.BPRLoss = loss_class
    student.train()
    aver_loss = 0.
    dns_time = 0.
    S,sam_time = sampler.PerSample()
    print(f"Logits[pre-sample][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2:]).long()
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    users, posItems, negItems = utils.TO(users, posItems, negItems)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    for (batch_i, 
         (batch_users, 
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        (batch_neg, weights, KD_loss, sam_time), time = timer(sampler.Sample, batch_users, batch_pos, batch_neg)
        dns_time += time
        weight_mean = torch.mean(weights).item()
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg, add_loss=KD_loss, weights=weights)        
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
            w.add_scalar(f'Weights/mean_weights', weight_mean, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    print(f"DNS[{dns_time}]")
    return f"BPR[aver loss{aver_loss:.3e}]"   
        
        
        
# ******************************************************************************
# ============================================================================**
# ============================================================================**
# ******************************************************************************
# TEST
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
    
def test_one_batch_ONE(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel_ONE(groundTrue, sorted_items)
    ndcg, hr= [], []
    for k in world.topks:
        ndcg.append(utils.NDCGatK_r_ONE(r, k))
        hr.append(utils.HRatK_ONE(r, k))
    return {'ndcg':np.array(ndcg), 
            'hr':np.array(hr)}
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.LightGCN
    # eval mode with no dropout
    Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    with torch.no_grad():
        users = list(testDict.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.DEVICE)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            if not world.TESTDATA:
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -1e5
            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if world.ONE:
            results = {'hr': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
            if multicore == 1:
                pre_results = pool.map(test_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(test_one_batch_ONE(x))
            scale = float(u_batch_size/len(users))
            for result in pre_results:
                results['hr'] += result['hr']
                results['ndcg'] += result['ndcg']
            results['hr'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            if w:
                w.add_scalars(f'Test/HR@{world.topks}',
                            {str(world.topks[i]): results['hr'][i] for i in range(len(world.topks))}, epoch)
                w.add_scalars(f'Test/NDCG@{world.topks}',
                            {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        else:
            results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
            if multicore == 1:
                pre_results = pool.map(test_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(test_one_batch(x))
            scale = float(u_batch_size/len(users))
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            if w:
                w.add_scalars(f'Test/Recall@{world.topks}',
                            {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
                w.add_scalars(f'Test/Precision@{world.topks}',
                            {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
                w.add_scalars(f'Test/NDCG@{world.topks}',
                            {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        return results
