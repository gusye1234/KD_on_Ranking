import world
import torch
import multiprocessing
from torch.nn import Softmax
import numpy as np
from time import time
from world import cprint
from model import PairWiseModel
from dataloader import BasicDataset

ALLPOS = None
# ----------------------------------------------------------------------------
# distill
class DistillSample:
    def __init__(self,
                 dataset : BasicDataset, 
                 student : PairWiseModel,
                 teacher : PairWiseModel,
                 dns_k : int,
                 method : int = 3,
                 beta = world.beta):
        """
            method 1 for convex combination
            method 2 for random indicator
            method 3 for simplified method 2
        """
        self.beta = beta
        self.W = torch.Tensor([world.p0])
        self.dataset = dataset
        self.student = student
        self.teacher = teacher
        self.methods = {
            'pass' : self.convex_combine, # not yet
            'indicator' : self.random_indicator,
            'simple' : self.max_min,
            'weight' : self.weight_pair
        }
        self.method = 'weight'
        self.Sample = self.methods[self.method]
        cprint(f"Using {self.method}")
        # self.Sample = self.max_min
        self.dns_k = dns_k
        self.start = False
        self.start_epoch = world.startepoch
        self.T = world.T
        self.soft = Softmax(dim=1)
        self.scale = 1
        self.t1 = 1
        self.t2 = 2.5

    def UniformSample_DNS_batch(self, epoch, batch_score_size=512):
        with torch.no_grad():
            if epoch >= self.start_epoch:
                self.start = True
            if self.start:
                print(">>W now:", self.W)
            else:
                print(">>DNS now")
            total_start = time()
            dataset = self.dataset
            dns_k = self.dns_k
            user_num = dataset.trainDataSize
            per_user_num = user_num // dataset.n_users + 1
            allPos = dataset.allPos
            S = []
            NEG_scores = []
            NEG_scores_teacher = []
            sample_time1 = 0.
            sample_time2 = 0.
            sample_time3 = 0.
            sample_time4 = 0.
            BinForUser = np.zeros(shape = (dataset.m_items, )).astype("int")
            # sample_shape = int(dns_k*1.5)+1
            BATCH_SCORE = None
            BATCH_SCORE_teacher = None
            now = 0
            NEG = np.zeros((per_user_num*dataset.n_users, dns_k))
            STUDENT = torch.zeros((per_user_num*dataset.n_users, dns_k))
            TEACHER = torch.zeros((per_user_num*dataset.n_users, dns_k))
            for user in range(dataset.n_users):
                start1 = time()
                if now >= batch_score_size:
                    del BATCH_SCORE
                    BATCH_SCORE = None
                    BATCH_SCORE_teacher = None
                if BATCH_SCORE is None:
                    left_limit = user+batch_score_size
                    batch_list = torch.arange(user, left_limit) if left_limit <= dataset.n_users else torch.arange(user, dataset.n_users)
                    BATCH_SCORE = self.student.getUsersRating(batch_list).cpu()
                    print(type(BATCH_SCORE), BATCH_SCORE.size())
                    # BATCH_SCORE_teacher = self.teacher.getUsersRating(batch_list, t1=self.t1, t2=self.t2)
                    now = 0
                sample_time1 += time()-start1

                start2 = time()
                scoreForuser = BATCH_SCORE[now]
                # scoreForuser_teacher = BATCH_SCORE_teacher[now] 
                scoreForuser_teacher = BATCH_SCORE[now]
                now += 1
                posForUser = allPos[user]
                if len(posForUser) == 0:
                    continue
                BinForUser[:] = 0
                BinForUser[posForUser] = 1
                NEGforUser = np.where(BinForUser == 0)[0]
                for i in range(per_user_num):
                    start3 = time()
                    posindex = np.random.randint(0, len(posForUser))
                    positem = posForUser[posindex]
                    negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
                    negitems = NEGforUser[negindex]
                    add_pair = (user, positem)
                    # NEG_scores.append(scoreForuser[negitems])
                    STUDENT[user*per_user_num + i, :] = scoreForuser[negitems]
                    TEACHER[user*per_user_num + i, :] = scoreForuser_teacher[negitems]
                    # NEG_scores_teacher.append(scoreForuser_teacher[negitems])

                    sample_time3 += time()-start3
                    start4 = time()
                    S.append(add_pair)
                    NEG[user*per_user_num + i, :] = negitems
                    sample_time4 += time() - start4
                sample_time2 += time() - start2
        # ===========================
        if self.start:
            self.W *= self.beta
        # return torch.Tensor(S), torch.from_numpy(NEG), torch.stack(NEG_scores), torch.stack(NEG_scores_teacher),[time() - total_start, sample_time1, sample_time2, sample_time3, sample_time4]        
        return torch.Tensor(S), torch.from_numpy(NEG), STUDENT, TEACHER,[time() - total_start, sample_time1, sample_time2, sample_time3, sample_time4]        
    # ----------------------------------------------------------------------------
    # method 1
    def convex_combine(self, batch_neg, student_score, teacher_score):
        pass
    # ----------------------------------------------------------------------------
    # method 4
    def weight_pair(self, batch_neg, batch_pos, batch_users, student_score, teacher_score):
        with torch.no_grad():
            start = time()
            if self.start:
                batch_list = torch.arange(0, len(batch_neg))
                _, student_max = torch.max(student_score, dim=1)
                # weights = teacher_score[batch_list, student_max]
                # weights = (1-weights)
                Items = batch_neg[batch_list, student_max]
                weights = self.teacher.pair_score(batch_users, batch_pos, Items)
                weights = self.scale*weights
                # print(torch.mean(weights))s
                return Items, weights, [time()-start, 0, 0]
            else:
                return self.DNS(batch_neg, student_score), None,[time()-start, 0, 0]
    # ----------------------------------------------------------------------------
    # method 2
    def random_indicator(self, batch_neg, batch_pos, batch_users, student_score, teacher_score):
        start = time()
        if self.start:
            batch_list = torch.arange(0, len(batch_neg))
            _, student_max = torch.max(student_score, dim=1)
            teacher_p = self.soft(-teacher_score/self.T)
            teacher_index = torch.multinomial(teacher_p, 1).squeeze()
            student_neg = batch_neg[batch_list, student_max]
            teacher_neg = batch_neg[batch_list, teacher_index]
            Items = torch.zeros((len(batch_neg), )).to(world.device).long()
            P_bern = torch.ones((len(batch_neg), ))*self.W
            indicator = torch.bernoulli(P_bern).bool()
            Items[indicator] = student_neg[indicator]
            Items[~indicator] = teacher_neg[~indicator]
            return Items, None,[time()-start, 0, 0]
        else:
            return self.DNS(batch_neg, student_score), None,[time()-start, 0, 0]
    # ----------------------------------------------------------------------------
    # method 3
    def max_min(self, batch_neg, batch_pos, batch_users, student_score, teacher_score):
        start = time()
        if self.start:
            batch_list = torch.arange(0, len(batch_neg))
            _, student_max = torch.max(student_score, dim=1)
            _, teacher_min = torch.min(teacher_score, dim=1)
            student_neg = batch_neg[batch_list, student_max]
            teacher_neg = batch_neg[batch_list, teacher_min]
            Items = torch.zeros((len(batch_neg), )).to(world.device).long()
            P_bern = torch.ones((len(batch_neg), ))*self.W
            indicator = torch.bernoulli(P_bern).bool()
            Items[indicator] = student_neg[indicator]
            Items[~indicator] = teacher_neg[~indicator]
            return Items, None,[time()-start, 0, 0]
        else:
            return self.DNS(batch_neg, student_score), None,[time()-start, 0, 0]
    # ----------------------------------------------------------------------------
    # just DNS
    def DNS(self, batch_neg, scores):
        batch_list = torch.arange(0, len(batch_neg))
        _, student_max = torch.max(scores, dim=1)
        student_neg = batch_neg[batch_list, student_max]
        return student_neg

# ----------------------------------------------------------------------------
# uniform sample
def UniformSample_original(users, dataset):
    """
    the original implement of BPR Sampling in LightGCN
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
# ----------------------------------------------------------------------------
# Dns sampling
def UniformSample_DNS_deter(users, dataset, dns_k):
    """
    the original implement of BPR Sampling in LightGCN
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
# ----------------------------------------------------------------------------
# batch rating for Dns sampling
def UniformSample_DNS_batch(users, dataset, model, dns_k, batch_score_size = 256):
    """
    the original implement of BPR Sampling in LightGCN
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
        sample_time1 = 0.
        sample_time2 = 0.
        BinForUser = np.zeros(shape = (dataset.m_items, )).astype("int")
        sample_shape = int(dns_k*1.5)+1
        BATCH_SCORE = None
        NEG = np.zeros((per_user_num*dataset.n_users, dns_k))
        SCORES = torch.zeros((per_user_num*dataset.n_users, dns_k))
        now = 0
        for user in range(dataset.n_users):
            start1 = time()
            if now >= batch_score_size:
                del BATCH_SCORE
                BATCH_SCORE = None
            if BATCH_SCORE is None:
                left_limit = user+batch_score_size
                batch_list = torch.arange(user, left_limit) if left_limit <= dataset.n_users else torch.arange(user, dataset.n_users)
                BATCH_SCORE = model.getUsersRating(batch_list).cpu()
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
            NEGforUser = np.where(BinForUser == 0)[0]
            for i in range(per_user_num):
                posindex = np.random.randint(0, len(posForUser))
                positem = posForUser[posindex]
                negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
                negitems = NEGforUser[negindex]
                add_pair = [user, positem]
                NEG[user*per_user_num + i, :] = negitems
                SCORES[user*per_user_num + i, :] = scoreForuser[negitems]
                S.append(add_pair)
            sample_time2 += time() - start2
    return torch.Tensor(S), torch.from_numpy(NEG),SCORES,[time() - total_start, sample_time1, sample_time2]

def DNS_sampling_batch(batch_neg, batch_score):
    start = time()
    batch_list = torch.arange(0, len(batch_neg))
    _, index = torch.max(batch_score, dim=1)
    return batch_neg[batch_list, index], [time()-start, 0, 0]
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