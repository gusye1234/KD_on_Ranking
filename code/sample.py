import world
import torch
import multiprocessing
import numpy as np
from torch.nn.functional import softplus
from time import time
from utils import Timer, shapes, combinations
from world import cprint
from model import PairWiseModel
from dataloader import BasicDataset
from torch.nn import Softmax, Sigmoid

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
    # method 5
    def oriKD(self, batch_neg, batch_pos, batch_users, student_score, teacher_score):
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
                weights = weights
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
            Items = torch.zeros((len(batch_neg), )).to(world.DEVICE).long()
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
            Items = torch.zeros((len(batch_neg), )).to(world.DEVICE).long()
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


class LogitsSample:
    def __init__(self,
                 dataset : BasicDataset, 
                 student : PairWiseModel,
                 teacher : PairWiseModel,
                 dns_k : int,
                 method : int = 3,
                 beta = world.beta):
        self.dataset = dataset
        self.student = student
        self.teacher = teacher
        self.beta = beta
        self.dns_k = dns_k
        # self.Sample = self.logits
        self.Sample = self.ranking
        self.sigmoid = Sigmoid()
        self.t = 0.3
        self.pairs = combinations(0, dns_k)
        world.cprint("======LOGITS baby====")
        
    def PerSample(self):
        return UniformSample_DNS_deter(self.dataset, self.dns_k)
    
    # ----------------------------------------------------------------------------
    # trivial method
    def logits(self, batch_users, batch_pos, batch_neg):
        """
        with grad
        """
        STUDENT = self.student
        TEACHER = self.teacher
        dns_k = self.dns_k
        
        NegItems = batch_neg
        negitem_vector = NegItems.reshape((-1, )) # dns_k * |users|
        user_vector = batch_users.repeat((dns_k, 1)).t().reshape((-1,))
        
        student_scores = STUDENT(user_vector, negitem_vector)
        student_scores = student_scores.reshape((-1, dns_k))
        student_pos_scores = STUDENT(batch_users, batch_pos)
        
        teacher_scores = TEACHER(user_vector, negitem_vector)
        teacher_scores = teacher_scores.reshape((-1, dns_k))
        teacher_pos_scores = TEACHER(batch_users, batch_pos)
        
        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(batch_users))
        negitems = NegItems[idx, top1]
        weights = self.sigmoid((teacher_pos_scores - teacher_scores[idx, top1])/self.t)
        
        KD_loss = self.beta*(1/2)*(
            (student_pos_scores - student_scores.t()) - (teacher_pos_scores - teacher_scores.t())
            ).norm(2).pow(2)
        KD_loss = KD_loss/float(len(batch_users))
        return negitems, weights, KD_loss, [0,0,0]
    # ----------------------------------------------------------------------------
    # ranking 
    def ranking(self, batch_users, batch_pos, batch_neg):
        """
        with grad
        """
        STUDENT = self.student
        TEACHER = self.teacher
        dns_k = self.dns_k
        times = []
        
        with Timer(times):    
            NegItems = batch_neg
            negitem_vector = NegItems.reshape((-1, )) # dns_k * |users|
            user_vector = batch_users.repeat((dns_k, 1)).t().reshape((-1,))
            
            student_scores = STUDENT(user_vector, negitem_vector)
            student_scores = student_scores.reshape((-1, dns_k))
            
            teacher_scores = TEACHER(user_vector, negitem_vector)
            teacher_scores = teacher_scores.reshape((-1, dns_k))
            teacher_pos_scores = TEACHER(batch_users, batch_pos)
            
            _, top1 = student_scores.max(dim=1)
            idx = torch.arange(len(batch_users))
            negitems = NegItems[idx, top1]
            weights = self.sigmoid((teacher_pos_scores - teacher_scores[idx, top1])/self.t)
        
        all_pairs = self.pairs.T
        pairs = self.pairs.T
        rank_loss = torch.tensor(0.)
        total_err = 0
        
        with Timer(times):
            for i, user in enumerate(batch_users) :
                # pairs = all_pairs[:, np.random.randint(all_pairs.shape[1], size=(8, ))]
                teacher_rank = (teacher_scores[i][pairs[0]] > teacher_scores[i][pairs[1]])
                student_rank = (student_scores[i][pairs[0]] > student_scores[i][pairs[1]])
                err_rank = torch.logical_xor(teacher_rank, student_rank)
                total_err += torch.sum(err_rank)
                should_rank_g = torch.zeros_like(teacher_rank).bool()
                should_rank_l = torch.zeros_like(teacher_rank).bool()
                # use teacher to confirm wrong rank
                should_rank_g[err_rank] = teacher_rank[err_rank]
                should_rank_l[err_rank] = (~teacher_rank)[err_rank]
                if torch.any(should_rank_g):
                    rank_loss += torch.mean(softplus(
                        (student_scores[i][pairs[1]] - student_scores[i][pairs[0]])[should_rank_g]
                    ))# should rank it higher
                if torch.any(should_rank_l):
                    rank_loss += torch.mean(softplus(
                        (student_scores[i][pairs[0]] - student_scores[i][pairs[1]])[should_rank_l]
                    ))# should rank it lower
                if torch.isnan(rank_loss) or torch.isinf(rank_loss):
                    print("student", student_scores[i])
                    print("pos", (student_scores[i][pairs[1]] - student_scores[i][pairs[0]])[should_rank_g])
                    print("neg", (student_scores[i][pairs[0]] - student_scores[i][pairs[1]])[should_rank_l])
                    exit(0)
        rank_loss /= len(batch_users)
        rank_loss = rank_loss*self.beta
        print(total_err, rank_loss)
        return negitems, weights, rank_loss, np.asanyarray(times)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
# ==============================================================
# NON-EXPERIMENTAL PART                                        =
# ==============================================================
    
# ----------------------------------------------------------------------------
# uniform sample
def UniformSample_original(users, dataset):
    """
    the original implement of BPR Sampling in LightGCN
    NOTE: we sample a whole epoch data at one time
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
def UniformSample_DNS_deter(dataset, dns_k):
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
        NEGforUser = np.where(BinForUser == 0)[0] 
        for i in range(per_user_num):
            # posindex = np.random.randint(0, len(posForUser))
            # positem = posForUser[posindex]
            # while True:
            #     negitems = np.random.randint(0, dataset.m_items, size=(dns_k, ))
            #     if np.sum(BinForUser[negitems]) > 0:
            #         continue
            #     else:
            #         break
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
            negitems = NEGforUser[negindex]
            add_pair = [user, positem]
            # NEG[user*per_user_num + i, :] = negitems
            add_pair = [user, positem, *negitems]
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

        _, top1 = scores.max(dim=1)
        idx = torch.arange(len(batch_users)).to(world.DEVICE)
        negitems = NegItems[idx, top1]
        sam_time2 = time() - sam_time2
    return negitems, [time() - start, 0, sam_time2]
# ----------------------------------------------------------------------------
# batch rating for Dns sampling
def UniformSample_DNS_batch(dataset, model, dns_k, batch_score_size = 256):
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