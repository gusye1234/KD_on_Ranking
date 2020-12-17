'''
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
from pprint import pprint
from sample import DistillSample
# from sample import DistillLogits
from model import PairWiseModel, BasicModel
from sample import UniformSample_DNS
from sample import UniformSample_original,DNS_sampling_neg
from utils import time2str, timer

item_count = None

CORES = multiprocessing.cpu_count() // 2


def Distill_DNS_yield(dataset, student, sampler, loss_class, epoch , w=None):
    """Batch version of Distill_DNS, using a generator to offer samples
    """
    sampler : DistillLogits
    bpr: utils.BPRLoss = loss_class
    student.train()
    aver_loss = 0
    with timer(name='sampling'):
        S = sampler.PerSample(batch=world.config['bpr_batch_size'])
    total_batch = dataset.trainDataSize // world.config['bpr_batch_size'] + 1
    for batch_i, Pairs in enumerate(S):
        Pairs = torch.from_numpy(Pairs).long().to(world.DEVICE)
        # print(Pairs.shape)
        batch_users, batch_pos, batch_neg = Pairs[:, 0], Pairs[:, 1], Pairs[:, 2:]
        with timer(name="KD"):
            batch_neg, weights, KD_loss = sampler.Sample(batch_users, batch_pos, batch_neg, epoch)
        with timer(name="BP"):
            cri = bpr.stageOne(batch_users, batch_pos, batch_neg, add_loss=KD_loss, weights=weights)
        aver_loss += cri
        # Additional section------------------------
        #
        # ------------------------------------------
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(dataset.trainDataSize / world.config['bpr_batch_size']) + batch_i)
        del Pairs
    aver_loss = aver_loss / total_batch
    info = f"{timer.dict()}[BPR loss{aver_loss:.3e}]"
    timer.zero()
    return info

def Distill_DNS(dataset, student, sampler, loss_class, epoch, w=None):
    """Training procedure for distillation methods

    Args:
        dataset (BasicDatset): defined in dataloader.BasicDataset, loaded in register.py
        student (PairWiseModel): recommend model with small dim
        sampler (DS|DL|RD|CD): tons of distill methods defined in sample.py
        loss_class (utils.BPRLoss): class to get BPR training loss, and BackPropagation
        epoch (int): 
        w (SummaryWriter, optional): Tensorboard writer

    Returns:
        str: summary of aver loss and running time for one epoch
    """
    sampler : DistillLogits
    bpr: utils.BPRLoss = loss_class
    student.train()
    aver_loss = 0
    with timer(name='sampling'):
        S = sampler.PerSample()
    S = torch.from_numpy(S).long().to(world.DEVICE)
    users, posItems, negItems = S[:, 0], S[:, 1], S[:, 2:]
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        with timer(name="KD"):
            batch_neg, weights, KD_loss = sampler.Sample(batch_users, batch_pos, batch_neg, epoch)
        with timer(name="BP"):
            cri = bpr.stageOne(batch_users, batch_pos, batch_neg, add_loss=KD_loss, weights=weights)
        aver_loss += cri
        # Additional section------------------------
        #
        # ------------------------------------------
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    info = f"{timer.dict()}[BPR loss{aver_loss:.3e}]"
    timer.zero()
    return info


def BPR_train_DNS_neg(dataset, recommend_model, loss_class, epoch, w=None):
    """Traininf procedure for DNS algorithms 

    Args:
        dataset (BasicDatset): defined in dataloader.BasicDataset, loaded in register.py
        recommend_model (PairWiseModel): recommend model with small dim
        loss_class (utils.BPRLoss): class to get BPR training loss, and BackPropagation
        epoch (int): 
        w (SummaryWriter, optional): Tensorboard writer

    Returns:
        str: summary of aver loss and running time for one epoch
    """
    Recmodel: PairWiseModel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    S = UniformSample_DNS(dataset, world.DNS_K)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2:]).long()
    users = users.to(world.DEVICE)
    posItems = posItems.to(world.DEVICE)
    negItems = negItems.to(world.DEVICE)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
            utils.minibatch(users,
                            posItems,
                            negItems,
                            batch_size=world.config['bpr_batch_size'])):
        if world.ALLDATA:
            weights = utils.getTestweight(batch_users, batch_pos, dataset)
        else:
            weights = None
        batch_neg = DNS_sampling_neg(batch_users, batch_neg, dataset, Recmodel)
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg, weights=weights)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(
                f'BPRLoss/BPR', cri,
                epoch * int(len(users) / world.config['bpr_batch_size']) +
                batch_i)
    # print(f"DNS[sampling][{time()-DNS_time:.1f}={DNS_time1:.2f}+{DNS_time2:.2f}]")
    aver_loss = aver_loss / total_batch
    return f"[BPR[aver loss{aver_loss:.3e}]"


def BPR_train_original(dataset, recommend_model, loss_class, epoch, w=None):
    """Traininf procedure for uniform BPR

    Args:
        dataset (BasicDatset): defined in dataloader.BasicDataset, loaded in register.py
        recommend_model (PairWiseModel): recommend model with small dim
        loss_class (utils.BPRLoss): class to get BPR training loss, and BackPropagation
        epoch (int): 
        w (SummaryWriter, optional): Tensorboard writer

    Returns:
        str: summary of aver loss and running time for one epoch
    """
    global item_count
    if item_count is None:
        item_count = torch.zeros(dataset.m_items)
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    S = UniformSample_original(dataset)
    S = torch.from_numpy(S).long()
    # print(f"BPR[sample time][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = S[:, 0]
    posItems = S[:, 1]
    negItems = S[:, 2]

    users = users.to(world.DEVICE)
    posItems = posItems.to(world.DEVICE)
    negItems = negItems.to(world.DEVICE)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(
            utils.minibatch(users,
                            posItems,
                            negItems,
                            batch_size=world.config['bpr_batch_size'])):
        if world.ALLDATA:
            print(world.ALLDATA)
            weights = utils.getTestweight(batch_users, batch_pos, dataset)
        else:
            weights = None
        item_count[batch_neg] += 1
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg, weights=weights)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(
                f'BPRLoss/BPR', cri,
                epoch * int(len(users) / world.config['bpr_batch_size']) +
                batch_i)
    aver_loss = aver_loss / total_batch
    return f"[BPR[aver loss{aver_loss:.3e}]"



# ******************************************************************************
# ============================================================================**
# ============================================================================**
# ******************************************************************************
# TEST
def test_one_batch(X):
    """helper function for Test
    """
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
    """helper function for Test, customized for leave-one-out dataset
    """
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel_ONE(groundTrue, sorted_items)
    ndcg, hr= [], []
    for k in world.topks:
        ndcg.append(utils.NDCGatK_r_ONE(r, k))
        hr.append(utils.HRatK_ONE(r, k))
    return {'ndcg':np.array(ndcg),
            'hr':np.array(hr)}

def Test(dataset, Recmodel, epoch, w=None, multicore=0, valid=True):
    """evaluate models

    Args:
        dataset (BasicDatset): defined in dataloader.BasicDataset, loaded in register.py
        Recmodel (PairWiseModel):
        epoch (int): 
        w (SummaryWriter, optional): Tensorboard writer
        multicore (int, optional): The num of cpu cores for testing. Defaults to 0.

    Returns:
        dict: summary of metrics
    """
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict
    if valid:
        testDict = dataset.validDict
    else:
        testDict = dataset.testDict
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
                w.add_scalars(f'Valid/HR@{world.topks}',
                            {str(world.topks[i]): results['hr'][i] for i in range(len(world.topks))}, epoch)
                w.add_scalars(f'Valid/NDCG@{world.topks}',
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
                w.add_scalars(f'Valid/Recall@{world.topks}',
                            {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
                w.add_scalars(f'Valid/Precision@{world.topks}',
                            {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
                w.add_scalars(f'Valid/NDCG@{world.topks}',
                            {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        return results


def Popularity_Bias(dataset, Recmodel, valid=True):

    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict
    Popularity = np.zeros(dataset.m_items, ).astype('int')
    if valid:
        testDict = dataset.validDict
    else:
        testDict = dataset.testDict
    Recmodel: model.LightGCN
    perUser = int(dataset.trainDataSize / dataset.n_users)
    # eval mode with no dropout
    Recmodel.eval()
    max_K = perUser
    user_topk = np.zeros((dataset.n_users, max_K), dtype=int)
    with torch.no_grad():
        users = list(testDict.keys())
        rating_list = []
        # ratings = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.DEVICE)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            if not world.TESTDATA:
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -1e5
            _, rating_K = torch.topk(rating, k=max_K)
            del rating
            rating_K = rating_K.numpy().astype('int')
            user_topk[batch_users] = rating_K
            for i in range(len(batch_users)):
                Popularity[rating_K[i]] +=1
    return Popularity.astype('int')
