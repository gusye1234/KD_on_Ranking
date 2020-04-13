import world
import torch
import numpy as np
import model
from dataloader import BasicDataset
from time import time


def UniformSample_original(users, dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    NOTE: we can sample a whole epoch data at one time
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
        sample_time1 += time() - start
    total = time() - total_start
    return np.array(S), [total, sample_time1, sample_time2]

def UniformSample_DNS(batch_users, dataset):
    """
    DNS pre-sample, only sample pos, for epoch
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        S.append([user, positem])
        sample_time1 += time() - start
    total = time() - total_start
    return np.array(S), [total, sample_time1, sample_time2]

def DNS_sampling(batch_users, dataset, recmodel):
    start = time()
    sam_time1 = time()
    with torch.no_grad():
        recmodel : model.PairWiseModel
        dataset : BasicDataset
        cpu_users = batch_users.cpu().numpy()
        dns_k  = world.DNS_K
        NegItems = []
        allPos = dataset.allPos
        for i, user in enumerate(cpu_users):
            posForuser = allPos[user]
            BinForuser = np.zeros((dataset.m_items, )).astype('uint8')
            BinForuser[posForuser] = 1
            assert np.sum(BinForuser) == len(posForuser)
            negForUsers = []
            while len(negForUsers) < dns_k:
                negitem = np.random.randint(0, dataset.m_items)
                if BinForuser[negitem] == 1:
                    # print('get a pos, cast')
                    continue
                else:
                    negForUsers.append(negitem)
            NegItems.append(negForUsers)
        sam_time1 = time() - sam_time1
        sam_time2 = time()
        NegItems = np.array(NegItems)
        negitem_vector = NegItems.reshape((-1, )) # dns_k * |users|
        negitem_vector = torch.Tensor(negitem_vector).long().to(world.device)
        user_vector = batch_users.repeat((dns_k, 1)).t().reshape((-1,))
        scores = recmodel(user_vector, negitem_vector)
        scores = scores.reshape((-1, dns_k))
        negitem_vector = negitem_vector.reshape((-1, dns_k))
        _, top1 = scores.max(dim=1)
        idx = torch.arange(len(cpu_users)).to(world.device)
        negitems = negitem_vector[idx, top1]
        sam_time2 = time() - sam_time2
    return negitems, [time() - start, sam_time1, sam_time2]
        
    
    
    
    
    
    
    
    
    