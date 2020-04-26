import world
import torch
import numpy as np
from model import PairWiseModel
from dataloader import BasicDataset
from time import time
import multiprocessing

ALLPOS = None

# ==========================================
# ==========================================
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
                negitem = np.random.randint(0, dataset.m_items)
                if BinForUser[negitem] == 1:
                    continue
                else:
                    break
            S.append([user, positem, negitem])
    return np.array(S), [time() - total_start, 0., 0.]
# ==========================================
# ==========================================
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

def UniformSample_DNS_batch(users, dataset, model, dns_k, batch_score_size = 2048):
    """
    the original impliment of BPR Sampling in LightGCN
    NOTE: we can sample a whole epoch data at one time
    :return:
        np.array
    """
    with torch.no_grad():
        total_start = time()
        dataset : BasicDataset
        model : PairWiseModel
        user_num = dataset.trainDataSize
        per_user_num = user_num // dataset.n_users + 1
        allPos = dataset.allPos
        S = []
        NEG_scores = []
        sample_time1 = 0.
        sample_time2 = 0.
        BinForUser = np.zeros(shape = (dataset.m_items, )).astype("int")
        BATCH_SCORE = None
        now = 0
        for user in range(dataset.n_users):
            start1 = time()
            if now >= batch_score_size:
                del BATCH_SCORE
                BATCH_SCORE = None
            if BATCH_SCORE is None:
                left_limit = user+batch_score_size
                batch_list = torch.arange(user, left_limit) if left_limit <= dataset.n_users else torch.arange(user, dataset.n_users)
                BATCH_SCORE = model.getUsersRating(batch_list).cpu().numpy()
                now = 0
            sample_time1 += time()-start1
            start2 = time()
            scoreForuser = BATCH_SCORE[now] 
            now += 1
            posForUser = allPos[user]
            if len(posForUser) == 0:
                continue
            BinForUser[:] = 0
            BinForUser[posForUser] = 1
            for i in range(per_user_num):
                posindex = np.random.randint(0, len(posForUser))
                positem = posForUser[posindex]
                while True:
                    negitems = np.random.randint(0, dataset.m_items, size=(dns_k*2, ))
                    itemIndex = BinForUser[negitems]
                    negInOne = negitems[itemIndex == 0]
                    if len(negInOne) < dns_k:
                        continue
                    else:
                        negitems = negitems[:dns_k]
                        break
                    # if np.sum(BinForUser[negitems]) > 0:
                    #     continue
                    # else:
                    #     break
                add_pair = [user, positem]
                add_pair.extend(negitems)
                NEG_scores.append(scoreForuser[negitems])
                S.append(add_pair)
            sample_time2 += time() - start2
    return np.array(S), np.array(NEG_scores),[time() - total_start, sample_time1, sample_time2]


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

def DNS_sampling_batch(batch_neg, batch_score):
    start = time()
    batch_list = torch.arange(0, len(batch_neg))
    _, index = torch.max(batch_score, dim=1)
    return batch_neg[batch_list, index], [time()-start, 0, 0]
# ==========================================
# ==========================================

# ============================================================================
# ============================================================================
# multi-core sampling, not yet
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