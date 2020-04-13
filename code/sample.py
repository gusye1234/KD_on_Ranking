import world
import torch
import numpy as np
import model
from dataloader import BasicDataset
from time import time
import multiprocessing

ALLPOS = None

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


def UniformSample_DNS_neg(users, dataset, dns_k):
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
    negItems = []
    sample_time1 = 0.
    sample_time2 = 0.
    BinForUser = np.zeros(shape = (dataset.m_items, )).astype("int")
    for i, user in enumerate(users):
        start = time()
        BinForUser[:] = 0
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        BinForUser[posForUser] = 1
        
        sample_time2 += time() - start
        start = time()
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        negitems = []
        while True:
            negitems = np.random.randint(0, dataset.m_items, size=(dns_k, ))
            if np.sum(BinForUser[negitems]) > 0:
                continue
            else:
                break
        add_pair = [user, positem]
        add_pair.extend(negitems)
        S.append(add_pair)
        sample_time1 += time() - start
    total = time() - total_start
    return np.array(S), [total, sample_time1, sample_time2]

def UniformSample_DNS_neg_multi(users, dataset, dns_k):
    """
    the original impliment of BPR Sampling in LightGCN
    NOTE: we can sample a whole epoch data at one time
    :return:
        np.array
    """
    global ALLPOS
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    ALLPOS = dataset.allPos
    S = []
    negItems = []
    sample_time1 = 0.
    sample_time2 = 0.
    pool = multiprocessing.Pool(world.CORES)
    dns_ks = [dns_k]*user_num
    m_itemss = [dataset.m_items]*user_num
    X = zip(users, m_itemss, dns_ks)
    results = pool.map(UniformSample_user, X)
    results = [data for data in results if data is not None]
    S = np.vstack(results)
    total = time() - total_start
    return S, [total, sample_time1, sample_time2]

def UniformSample_user(X):
    user = X[0]
    m_items = X[1]
    dns_k = X[2]
    posForUser = ALLPOS[user]
    BinForUser = np.zeros(shape = (m_items, )).astype("int")
    if len(posForUser) == 0:
        return None
    BinForUser[posForUser] = 1
    start = time()
    posindex = np.random.randint(0, len(posForUser))
    positem = posForUser[posindex]
    negitems = []
    while True:
        negitems = np.random.randint(0, m_items, size=(dns_k, ))
        if np.sum(BinForUser[negitems]) > 0:
            continue
        else:
            break
    add_pair = [user, positem]
    add_pair.extend(negitems)
    return np.array(add_pair).astype('int')

def DNS_sampling_neg(batch_users, batch_neg, dataset, recmodel):
    start = time()
    sam_time1 = time()
    dns_k = world.DNS_K
    with torch.no_grad():
        sam_time2 = time()
        NegItems = batch_neg
        negitem_vector = NegItems.reshape((-1, )) # dns_k * |users|
        user_vector = batch_users.repeat((dns_k, 1)).t().reshape((-1,))
        scores = recmodel(user_vector, negitem_vector)
        scores = scores.reshape((-1, dns_k))
        negitem_vector = negitem_vector.reshape((-1, dns_k))
        _, top1 = scores.max(dim=1)
        idx = torch.arange(len(batch_users)).to(world.device)
        negitems = negitem_vector[idx, top1]
        sam_time2 = time() - sam_time2
    return negitems, [time() - start, 0, sam_time2]

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
        BinForuser = np.zeros((dataset.m_items, )).astype('uint8')
        for i, user in enumerate(cpu_users):
            BinForuser[:] = 0
            posForuser = allPos[user]
            BinForuser[posForuser] = 1
            assert np.sum(BinForuser) == len(posForuser)
            negForUsers = []
            while True:
                negForUsers = np.random.randint(0, dataset.m_items, size=(dns_k, ))
                if np.sum(BinForuser[negForUsers]) > 0:
                    continue
                else:
                    break
            # while len(negForUsers) < dns_k:
            #     negitem = np.random.randint(0, dataset.m_items)
            #     if BinForuser[negitem] == 1:
            #         # print('get a pos, cast')
            #         continue
            #     else:
            #         negForUsers.append(negitem)
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
        

def UniformSample_DNS_deter(users, dataset, dns_k):
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
    per_user_num = user_num // dataset.n_users + 1
    allPos = dataset.allPos
    S = []
    negItems = []
    sample_time1 = 0.
    sample_time2 = 0.
    BinForUser = np.zeros(shape = (dataset.m_items, )).astype("int")
    for user in range(dataset.n_users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        BinForUser[:] = 0
        BinForUser[posForUser] = 1
        for i in range(per_user_num):
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            while True:
                negitems = np.random.randint(0, dataset.m_items, size=(dns_k, ))
                if np.sum(BinForUser[negitems]) > 0:
                    continue
                else:
                    break
            add_pair = [user, positem]
            add_pair.extend(negitems)
            S.append(add_pair)
    return np.array(S), [time() - total_start, 0., 0.]
    
    
    
    
    
    
    