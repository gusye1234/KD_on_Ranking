import world
import torch
import multiprocessing
import numpy as np
from torch.nn.functional import softplus
from time import time
from utils import Timer, shapes, combinations, timer
from world import cprint
from model import PairWiseModel
from dataloader import BasicDataset
from torch.nn import Softmax, Sigmoid
import torch.nn.functional as F
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sample_ext = True
except:
    sample_ext = False

ALLPOS = None
# ----------------------------------------------------------------------------
# distill


def userAndMatrix(batch_users, batch_items, model):
    """cal scores between user vector and item matrix

    Args:
        batch_users (tensor): vector (batch_size)
        batch_items (tensor): matrix (batch_size, dim_item)
        model (Module):

    Returns:
        tensor: scores, shape like batch_items
    """
    dim_item = batch_items.shape[-1]
    vector_user = batch_users.repeat((dim_item, 1)).t().reshape((-1, ))
    vector_item = batch_items.reshape((-1, ))
    return model(vector_user, vector_item).reshape((-1, dim_item))


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
        # self.methods = {
        #     'combine' : self.convex_combine, # not yet
        #     'indicator' : self.random_indicator,
        #     'simple' : self.max_min,
        #     'weight' : self.weight_pair,
        # }
        self.method = 'combine'
        self.Sample = self.convex_combine
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

    def PerSample(self):
        return UniformSample_DNS_deter(self.dataset, self.dns_k)

    # ----------------------------------------------------------------------------
    # method 1
    def convex_combine(self, batch_users, batch_pos, batch_neg, epoch):
        with torch.no_grad():
            student_score = userAndMatrix(batch_users, batch_neg, self.student)
            teacher_score = userAndMatrix(batch_users, batch_neg, self.teacher)
            start = time()
            batch_list = torch.arange(0, len(batch_neg))
            pos_score = self.teacher(batch_users, batch_pos).unsqueeze(dim=1)
            margin  = pos_score - teacher_score
            refine = margin*student_score
            _, student_max = torch.max(refine, dim=1)
            Items = batch_neg[batch_list, student_max]
            return Items, None, None
    # ----------------------------------------------------------------------------
    # method 4
    # def weight_pair(self, batch_users, batch_pos, batch_neg, epoch):
    #     with torch.no_grad():
    #         start = time()
    #         student_score = userAndMatrix(batch_users, batch_neg, self.student)
    #         teacher_score = userAndMatrix(batch_users, batch_neg, self.teacher)
    #         batch_list = torch.arange(0, len(batch_neg))
    #         _, student_max = torch.max(student_score, dim=1)
    #         Items = batch_neg[batch_list, student_max]
    #         weights = self.teacher.pair_score(batch_users, batch_pos, Items)
    #         weights = weights
    #         return Items, weights, None,[time()-start, 0, 0]
    # ----------------------------------------------------------------------------
    # method 2
    # def random_indicator(self, batch_users, batch_pos, batch_neg, epoch):
    #     start = time()
    #     if self.start:
    #         student_score = userAndMatrix(batch_users, batch_neg, self.student)
    #         teacher_score = userAndMatrix(batch_users, batch_neg, self.teacher)

    #         batch_list = torch.arange(0, len(batch_neg))
    #         _, student_max = torch.max(student_score, dim=1)
    #         teacher_p = self.soft(-teacher_score/self.T)
    #         teacher_index = torch.multinomial(teacher_p, 1).squeeze()
    #         student_neg = batch_neg[batch_list, student_max]
    #         teacher_neg = batch_neg[batch_list, teacher_index]
    #         Items = torch.zeros((len(batch_neg), )).to(world.DEVICE).long()
    #         P_bern = torch.ones((len(batch_neg), ))*self.W
    #         indicator = torch.bernoulli(P_bern).bool()
    #         Items[indicator] = student_neg[indicator]
    #         Items[~indicator] = teacher_neg[~indicator]
    #         return Items, None,[time()-start, 0, 0]
    #     else:
    #         return self.DNS(batch_neg, student_score), None,[time()-start, 0, 0]
    # ----------------------------------------------------------------------------
    # method 3
    # def max_min(self, batch_users, batch_pos, batch_neg, epoch):
    #     start = time()
    #     if self.start:
    #         student_score = userAndMatrix(batch_users, batch_neg, self.student)
    #         teacher_score = userAndMatrix(batch_users, batch_neg, self.teacher)
    #         batch_list = torch.arange(0, len(batch_neg))
    #         _, student_max = torch.max(student_score, dim=1)
    #         _, teacher_min = torch.min(teacher_score, dim=1)
    #         student_neg = batch_neg[batch_list, student_max]
    #         teacher_neg = batch_neg[batch_list, teacher_min]
    #         Items = torch.zeros((len(batch_neg), )).to(world.DEVICE).long()
    #         P_bern = torch.ones((len(batch_neg), ))*self.W
    #         indicator = torch.bernoulli(P_bern).bool()
    #         Items[indicator] = student_neg[indicator]
    #         Items[~indicator] = teacher_neg[~indicator]
    #         return Items, None,[time()-start, 0, 0]
    #     else:
    #         return self.DNS(batch_neg, student_score), None,[time()-start, 0, 0]
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
        self.Sample = self.logits
        # self.Sample = self.ranking
        self.sigmoid = Sigmoid()
        self.t = 0.3
        self.pairs = combinations(0, dns_k)
        world.cprint("======LOGITS baby====")

    def PerSample(self):
        return UniformSample_DNS_deter(self.dataset, self.dns_k)

    # ----------------------------------------------------------------------------
    # trivial method
    def logits(self, batch_users, batch_pos, batch_neg, epoch):
        """
        with grad
        """
        STUDENT = self.student
        TEACHER = self.teacher
        dns_k = self.dns_k

        student_scores = userAndMatrix(batch_users, batch_neg, STUDENT)
        student_pos_scores = STUDENT(batch_users, batch_pos)

        teacher_scores = userAndMatrix(batch_users, batch_neg, TEACHER)
        teacher_pos_scores = TEACHER(batch_users, batch_pos)

        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(batch_users))
        negitems = NegItems[idx, top1]
        weights = self.sigmoid((teacher_pos_scores - teacher_scores[idx, top1])/self.t)

        KD_loss = self.beta*(1/2)*(
            (student_pos_scores - student_scores.t()) - (teacher_pos_scores - teacher_scores.t())
            ).norm(2).pow(2)
        KD_loss = KD_loss/float(len(batch_users))
        return negitems, weights, KD_loss
    # ----------------------------------------------------------------------------
    # ranking
    '''
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
        rank_loss = torch.tensor(0.).to(world.DEVICE)
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
        print(f"{total_err.item()}-> ", end='')
        return negitems, weights, rank_loss, np.asanyarray(times)
    '''


class RD:
    def __init__(self,
                 dataset : BasicDataset,
                 student : PairWiseModel,
                 teacher : PairWiseModel,
                 dns,
                 mu,
                 topK,
                 lamda,
                 teach_alpha,
                 dynamic_sample = 100,
                 dynamic_start_epoch = None,
                 ):
        self.rank_aware = False
        self.dataset = dataset
        self.STUDENT = student
        self.TEACHER = teacher.eval()
        self.RANK = None
        self.epoch = 0

        self._weight_renormalize = False
        self.mu, self.topk, self.lamda = mu, topK, lamda
        self.teach_alpha = teach_alpha
        self.start_epoch = dynamic_start_epoch
        self._generateTopK()
        self._static_weights = self._generateStaticWeights()

    def PerSample(self):
        return UniformSample_DNS_deter(self.dataset, self.dns_k)

    def _generateStaticWeights(self):
        w = torch.arange(1, self.topk)
        w = torch.exp(-w/self.lamda)
        return (w/w.sum()).unsqueeze(0)

    def _generateTopK(self, batch_size = 256):
        if self.RANK is None:
            self.RANK = torch.zeros((self.dataset.n_users,self.topk)).to(world.DEVICE)
            for user in range(0, self.dataset.n_users, batch_size):
                scores = self.TEACHER.getUsersRating(torch.arange(user, user+batch_size))
                pos_item = self.dataset.getUserPosItems(np.arange(user, user+batch_size))

                # -----
                exclude_user, exclude_item = [], []
                for i, items in enumerate(pos_item):
                    exclude_user.extend([i]*len(items))
                    exclude_item.extend(items)
                scores[exclude_user, exclude_item] = -1e5
                # -----
                _, neg_item = torch.topk(scores, self.topk)
                self.RANK[user:user + batch_size] = neg_item

    def _rank_aware_weights(self):
        pass

    def _weights(self, S_score_in_T, epoch, dynamic_samples):
        if epoch < self.start_epoch:
            return self._static_weights.repeat((batch, 1))
        with torch.no_grad():
            batch = S_score_in_T.shape[0]
            static_weights = self._static_weights.repeat((batch, 1))
            # ---
            topk = teacher_rank.shape[-1]
            num_dynamic = dynamic_samples.shape[-1]
            m_items = self.dataset.m_items
            dynamic_weights = torch.zeros(batch, topk)
            for col in range(topk):
                col_prediction = S_score_in_T[:, col].unsqueeze(1)
                num_smaller    = torch.sum(col_prediction < dynamic_samples, dim=1).float()
                relative_rank  = num_smaller / num_dynamic
                appro_rank     = torch.floor((m_items - 1)*relative_rank)

                dynamic = torch.tanh(self.mu * (appro_rank - col))
                dynamic = torch.clamp(dynamic, min=0.)

                dynamic_weights[:, col] = dynamic.squeeze()
            if self._weight_renormalize:
                return F.normalize(static_weights*dynamic_weights, p=1, dim=1)
            else:
                return static_weights*dynamic_weights


    def Sample(self, batch_users, batch_pos, batch_neg, epoch, dynamic_samples=None):
        STUDENT = self.student
        TEACHER = self.teacher
        dns_k = self.dns_k
        # ----
        student_scores = userAndMatrix(batch_users, batch_neg, STUDENT)

        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(batch_users))
        negitems = batch_neg[idx, top1]
        # ----
        topk_teacher = self.RANK[batch_users]
        topk_teacher = topk_teacher.reshape((-1,))
        user_vector = batch_users.repeat((self.topk, 1)).t().reshape((-1, ))

        S_score_in_T = STUDENT(user_vector, topk_teacher)
        weights = self._weights(S_score_in_T.reshape((-1, self.topk)).detach(),
                                epoch,
                                dynamic_samples)
        weights = weights.reshape((-1,))
        # RD_loss
        RD_loss = weights*torch.log(torch.sigmoid(S_score_in_T))
        RD_loss = RD.sum(1)
        RD_loss = self.teach_alpha*RD_loss.mean()

        return negitems, None, RD_loss


class CD:
    def __init__(self,
                 dataset : BasicDataset,
                 student : PairWiseModel,
                 teacher : PairWiseModel,
                 dns,
                 lamda,
                 n_distill,
                 t1=None,
                 t2=None):
        self.student = student
        self.teacher = teacher.eval()
        self.dataset = dataset
        self.dns_k = dns

        self.lamda = lamda
        self.n_distill = n_distill
        self.t1, self.t2 = t1, t2

    def PerSample(self):
        return UniformSample_DNS_deter(self.dataset, self.dns_k)

    def random_sample(self, batch_size):
        samples = np.random.choice(self.dataset.m_items, (batch_size, self.n_distill))
        return torch.from_numpy(samples).to(world.DEVICE)

    def student_sample(self, batch_users):
        MODEL = self.student
        # uniform_sample =
        pass

    def teacher_sampler(self, batch_users):
        pass

    def base(self, batch_users, batch_pos, batch_neg, epoch):
        STUDENT = self.student
        TEACHER = self.teacher
        dns_k = self.dns_k
        # ----
        student_scores = userAndMatrix(batch_users, batch_neg, STUDENT)

        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(batch_users))
        negitems = batch_neg[idx, top1]
        # ----
        random_samples = self.random_sample(batch_users.shape[0])
        samples_vector = random_samples.reshape((-1, ))
        samples_scores_T = TEACHER(user_vector, samples_vector)
        samples_scores_S = STUDENT(user_vector, samples_vector)
        weights = torch.sigmoid((samples_scores_T + self.t2)/self.t1)
        inner = torch.sigmoid(samples_scores_S)
        CD_loss = -(
            weights*torch.log(inner + 1e-10) + \
            (1-weights)*torch.log(1 - inner + 1e-10)
        )
        return negitems, None, CD_loss

    def student_guide(self, batch_users, batch_pos, batch_neg, epoch):
        pass









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
        with timer(name="L"):
            BinForUser[posForUser] = 1
            NEGforUser = np.where(BinForUser == 0)[0]
        for i in range(per_user_num):
            with timer(name='index'):
                posindex = np.random.randint(0, len(posForUser))
                positem = posForUser[posindex]
                negindex = np.random.randint(0, len(NEGforUser))
                negitem = NEGforUser[negindex]
            with timer(name="append"):
                S.append([user, positem, negitem])
    print(timer.dict())
    timer.zero()
    return np.array(S), [time() - total_start, 0., 0.]
# ----------------------------------------------------------------------------
# Dns sampling
def UniformSample_DNS_deter(dataset, dns_k):
    """
    sample dns_k negative items for each user-pos pair
    """
    dataset : BasicDataset
    allPos = dataset.allPos
    if sample_ext:
        S = sampling.sample_negative(
            dataset.n_users,
            dataset.m_items,
            dataset.trainDataSize,
            allPos,
            dns_k
        )
    else:
        return UniformSample_DNS_deter_python(dataset, dns_k)
    return S

def UniformSample_DNS_deter_python(dataset, dns_k):
    """
    sample dns_k negative items for each user-pos pair
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    per_user_num = user_num // dataset.n_users + 1
    allPos = dataset.allPos
    S = []
    # S = torch.zeros(dataset.n_users*per_user_num, 2+dns_k)
    # negItems = []
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
            posindex = np.random.randint(0, len(posForUser))
            positem = posForUser[posindex]
            negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
            negitems = NEGforUser[negindex]
            add_pair = [user, positem, *negitems]
            S.append(add_pair)
    return S, [time() - total_start, 0., 0.]

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


if __name__ == "__main__":
    method = UniformSample_DNS_deter
    from register import dataset
    from utils import timer
    for i in range(10):
        with timer():
            method(dataset, 10)
        print(timer.get())