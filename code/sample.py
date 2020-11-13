import world
import torch
import multiprocessing
import numpy as np
from torch.nn.functional import softplus
from time import time
from utils import Timer, shapes, combinations, timer
from world import cprint
from model import PairWiseModel, LightGCN
from dataloader import BasicDataset
from torch.nn import Softmax, Sigmoid
import torch.nn.functional as F
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.SEED)
    sample_ext = True
except:
    world.cprint("Cpp ext not loaded")
    sample_ext = False

ALLPOS = None
# ----------------------------------------------------------------------------
# distill


def userAndMatrix(batch_users, batch_items, model):
    """cal scores between user vector and item matrix

    Args:
        batch_users (tensor): vector (batch_size)
        batch_items (tensor): matrix (batch_size, dim_item)
        model (PairWiseModel):

    Returns:
        tensor: scores, shape like batch_items
    """
    dim_item = batch_items.shape[-1]
    vector_user = batch_users.repeat((dim_item, 1)).t().reshape((-1, ))
    vector_item = batch_items.reshape((-1, ))
    return model(vector_user, vector_item).reshape((-1, dim_item)).to(world.DEVICE)

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

    def PerSample(self, batch=None):
        if batch is not None:
            return UniformSample_DNS_yield(self.dataset, self.dns_k, batch_size=batch)
        else:
            return UniformSample_DNS(self.dataset, self.dns_k)

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
    # just DNS
    def DNS(self, batch_neg, scores):
        batch_list = torch.arange(0, len(batch_neg))
        _, student_max = torch.max(scores, dim=1)
        student_neg = batch_neg[batch_list, student_max]
        return student_neg

class DistillLogits:
    def __init__(self,
                 dataset : BasicDataset,
                 student : PairWiseModel,
                 teacher : PairWiseModel,
                 dns_k : int,
                 beta = world.beta):
        self.dataset = dataset
        self.student = student
        self.teacher = teacher
        self.beta = beta
        self.dns_k = dns_k
        self.Sample = self.logits
        # self.Sample = self.ranking
        self.sigmoid = Sigmoid()
        self.t = 1
        self.pairs = combinations(0, dns_k)
        world.cprint("======LOGITS baby====")

    def PerSample(self, batch=None):
        if batch is not None:
            return UniformSample_DNS_yield(self.dataset,
                                           self.dns_k,
                                           batch_size=batch)
        else:
            return UniformSample_DNS(self.dataset, self.dns_k)

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
        negitems = batch_neg[idx, top1]
        weights = self.sigmoid((teacher_pos_scores - teacher_scores[idx, top1])/self.t)

        KD_loss = self.beta*(1/2)*(
            (student_pos_scores - student_scores.t()) - (teacher_pos_scores - teacher_scores.t())
            ).norm(2).pow(2)
        KD_loss = KD_loss/float(len(batch_users))
        return negitems, weights, KD_loss

class RD:
    def __init__(self,
                 dataset : BasicDataset,
                 student : PairWiseModel,
                 teacher : PairWiseModel,
                 dns,
                 topK=10,
                 mu=0.1,
                 lamda=1,
                 teach_alpha=1.0,
                 dynamic_sample = 100,
                 dynamic_start_epoch = 0,
                 ):
        self.rank_aware = False
        self.dataset = dataset
        self.student = student
        self.teacher = teacher.eval()
        self.RANK = None
        self.epoch = 0

        self._weight_renormalize = False
        self.mu, self.topk, self.lamda = mu, topK, lamda
        self.dynamic_sample_num = dynamic_sample
        self.dns_k = dns
        self.teach_alpha = teach_alpha
        self.start_epoch = dynamic_start_epoch
        self._generateTopK()
        self._static_weights = self._generateStaticWeights()

    def PerSample(self, batch=None):
        if batch is not None:
            return UniformSample_DNS_yield(self.dataset,
                                           self.dns_k+self.dynamic_sample_num,
                                           batch_size=batch)
        else:
            return UniformSample_DNS(self.dataset,
                                     self.dns_k+self.dynamic_sample_num)

    def _generateStaticWeights(self):
        w = torch.arange(1, self.topk+1).float()
        w = torch.exp(-w/self.lamda)
        return (w/w.sum()).unsqueeze(0)

    def _generateTopK(self, batch_size = 256):
        if self.RANK is None:
            with torch.no_grad():
                self.RANK = torch.zeros((self.dataset.n_users,self.topk)).to(world.DEVICE)
                for user in range(0, self.dataset.n_users, batch_size):
                    end = min(user+batch_size, self.dataset.n_users)
                    scores = self.teacher.getUsersRating(torch.arange(user, end))
                    pos_item = self.dataset.getUserPosItems(np.arange(user, end))

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

    def _weights(self, S_score_in_T, epoch, dynamic_scores):
        batch = S_score_in_T.shape[0]
        if epoch < self.start_epoch:
            return self._static_weights.repeat((batch, 1))
        with torch.no_grad():
            static_weights = self._static_weights.repeat((batch, 1))
            # ---
            topk = S_score_in_T.shape[-1]
            num_dynamic = dynamic_scores.shape[-1]
            m_items = self.dataset.m_items
            dynamic_weights = torch.zeros(batch, topk)
            for col in range(topk):
                col_prediction = S_score_in_T[:, col].unsqueeze(1)
                num_smaller    = torch.sum(col_prediction < dynamic_scores, dim=1).float()
                # print(num_smaller.shape)
                relative_rank  = num_smaller / num_dynamic
                appro_rank     = torch.floor((m_items - 1)*relative_rank)

                dynamic = torch.tanh(self.mu * (appro_rank - col))
                dynamic = torch.clamp(dynamic, min=0.)

                dynamic_weights[:, col] = dynamic.squeeze()
            if self._weight_renormalize:
                return F.normalize(static_weights*dynamic_weights, p=1, dim=1).to(world.DEVICE)
            else:
                return (static_weights*dynamic_weights).to(world.DEVICE)


    def Sample(self, batch_users, batch_pos, batch_neg, epoch, dynamic_samples=None):
        STUDENT = self.student
        TEACHER = self.teacher
        assert batch_neg.shape[-1] == (self.dns_k + self.dynamic_sample_num)
        dynamic_samples = batch_neg[:, -self.dynamic_sample_num:]
        batch_neg = batch_neg[:, :self.dns_k]
        # ----
        student_scores = userAndMatrix(batch_users, batch_neg, STUDENT)
        dynamic_scores = userAndMatrix(batch_users, dynamic_samples, STUDENT).detach()

        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(batch_users))
        negitems = batch_neg[idx, top1]
        # ----
        topk_teacher = self.RANK[batch_users]
        topk_teacher = topk_teacher.reshape((-1,)).long()
        user_vector = batch_users.repeat((self.topk, 1)).t().reshape((-1, ))

        S_score_in_T = STUDENT(user_vector, topk_teacher).reshape((-1, self.topk))
        weights = self._weights(S_score_in_T.detach(),
                                epoch,
                                dynamic_scores)
        # RD_loss
        RD_loss = weights*torch.log(torch.sigmoid(S_score_in_T))
        # print("RD shape", RD_loss.shape)
        RD_loss = RD_loss.sum(1)
        RD_loss = self.teach_alpha*RD_loss.mean()

        return negitems, None, RD_loss


class CD:
    def __init__(self,
                 dataset : BasicDataset,
                 student : PairWiseModel,
                 teacher : PairWiseModel,
                 dns,
                 lamda=0.5,
                 n_distill=50,
                 t1=1,
                 t2=0):
        self.student = student
        self.teacher = teacher.eval()
        self.dataset = dataset
        self.dns_k = dns

        self.strategy = "student guide"
        self.lamda = lamda
        self.n_distill = n_distill
        self.t1, self.t2 = t1, t2

    def PerSample(self, batch=None):
        if batch is not None:
            return UniformSample_DNS_yield(self.dataset,
                                           self.dns_k,
                                           batch_size=batch)
        else:
            return UniformSample_DNS(self.dataset,
                                     self.dns_k)

    def Sample(self, batch_users, batch_pos, batch_neg, epoch):
        return self.sample_diff(batch_users, batch_pos, batch_neg, self.strategy)

    def random_sample(self, batch_size):
        samples = np.random.choice(self.dataset.m_items, (batch_size, self.n_distill))
        return torch.from_numpy(samples).long().to(world.DEVICE)

    def rank_sample(self, batch_users, MODEL):
        if MODEL is None:
            return self.random_sample(len(batch_users))
        MODEL : LightGCN
        all_items = self.dataset.m_items
        batch_size = len(batch_users)
        rank_samples = torch.zeros(batch_size, self.n_distill)
        with torch.no_grad():
            items_score = MODEL.getUsersRating(batch_users)
            index = torch.arange(all_items).long()
            for i in range(batch_size):
                rating = items_score[i]
                while True:
                    random_index = torch.from_numpy(np.random.randint(all_items, size=(all_items,))).long()
                    compared = (rating[index] > rating[random_index])
                    if torch.sum(compared) < self.n_distill:
                        continue
                    else:
                        sampled_items = index[compared]
                        sampled_items = torch.topk(sampled_items, k=self.n_distill)[1]
                        break
                rank_samples[i] = sampled_items
        return rank_samples.to(world.DEVICE).long()

    def sample_diff(self, batch_users, batch_pos, batch_neg, strategy):
        STUDENT = self.student
        TEACHER = self.teacher
        if strategy == "random":
            MODEL = None
        elif strategy == "student guide":
            MODEL = STUDENT
        elif strategy == "teacher guide":
            MODEL = TEACHER
        else:
            raise TypeError("CD support [random, student guide, teacher guide], " \
                            f"But got {strategy}")
        dns_k = self.dns_k
        # ----
        student_scores = userAndMatrix(batch_users, batch_neg, STUDENT)

        _, top1 = student_scores.max(dim=1)
        idx = torch.arange(len(batch_users))
        negitems = batch_neg[idx, top1]
        # ----
        random_samples = self.rank_sample(batch_users, MODEL=MODEL)
        # samples_vector = random_samples.reshape((-1, ))
        samples_scores_T = userAndMatrix(batch_users, random_samples, TEACHER)
        samples_scores_S = userAndMatrix(batch_users, random_samples, STUDENT)
        weights = torch.sigmoid((samples_scores_T + self.t2)/self.t1)
        inner = torch.sigmoid(samples_scores_S)
        CD_loss = -(
            weights*torch.log(inner + 1e-10) + \
            (1-weights)*torch.log(1 - inner + 1e-10)
        )
        # print(CD_loss.shape)
        CD_loss = CD_loss.sum(1).mean()
        return negitems, None, CD_loss

    def student_guide(self, batch_users, batch_pos, batch_neg, epoch):
        pass



# ==============================================================
# NON-EXPERIMENTAL PART
# ==============================================================

# ----------
# uniform sample
def UniformSample_original(dataset):
    """
    the original implement of BPR Sampling in LightGCN
    NOTE: we sample a whole epoch data at one time
    :return:
        np.array
    """
    return UniformSample_DNS(dataset, 1)

# ----------
# Dns sampling
def UniformSample_DNS_yield(dataset,
                            dns_k,
                            batch_size=None):
    """Generate train samples(already shuffled)

    Args:
        dataset (BasicDataset)
        dns_k ([int]): How many neg samples for one (user,pos) pair
        batch_size ([int], optional) Defaults to world.config['bpr_batch_size'].

    Returns:
        [ndarray]: yield (batch_size, 2+dns_k)
    """
    dataset: BasicDataset
    batch_size = batch_size or world.config['bpr_batch_size']
    allPos = dataset.allPos
    All_users = np.random.randint(dataset.n_users,
                                  size=(dataset.trainDataSize, ))
    for batch_i in range(0, len(All_users), batch_size):
        batch_users = All_users[batch_i:batch_i+batch_size]
        if sample_ext:
            yield sampling.sample_negative_ByUser(
                batch_users, dataset.m_items, allPos, dns_k
            )
        else:
            yield UniformSample_DNS_python_ByUser(
                dataset, batch_users, dns_k
            )


def UniformSample_DNS(dataset, dns_k):
    """Generate train samples(sorted)

    Args:
        dataset ([BasicDataset])
        dns_k ([int]): How many neg samples for one (user,pos) pair

    Returns:
        [ndarray]: shape (The num of interactions, 2+dns_k)
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
        return S
    else:
        return UniformSample_DNS_python(dataset, dns_k)

def UniformSample_DNS_python(dataset, dns_k):
    """python implementation for 'UniformSample_DNS'
    """
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    per_user_num = user_num // dataset.n_users + 1
    allPos = dataset.allPos
    S = []
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
    return S


def UniformSample_DNS_python_ByUser(dataset, users, dns_k):
    """python implementation for 
    cpp ext 'sampling.sample_negative_ByUser' in sources/sampling.cpp
    """
    dataset: BasicDataset
    allPos = dataset.allPos
    S = np.zeros((len(users), 2+dns_k))
    BinForUser = np.zeros(shape=(dataset.m_items, )).astype("int")
    for i, user in enumerate(users):
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        BinForUser[:] = 0
        BinForUser[posForUser] = 1
        NEGforUser = np.where(BinForUser == 0)[0]
        negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
        negitems = NEGforUser[negindex]
        S[i] = [user, positem, *negitems]
    return S


def DNS_sampling_neg(batch_users, batch_neg, dataset, recmodel):
    """Dynamic Negative Choosing.(return max in Neg)

    Args:
        batch_users ([tensor]): shape (batch_size, )
        batch_neg ([tensor]): shape (batch_size, dns_k)
        dataset ([BasicDataset])
        recmodel ([PairWiseModel])

    Returns:
        [tensor]: Vector of negitems, shape (batch_size, ) 
                  corresponding to batch_users
    """
    dns_k = world.DNS_K
    with torch.no_grad():

        scores = userAndMatrix(batch_users, batch_neg, recmodel)

        _, top1 = scores.max(dim=1)
        idx = torch.arange(len(batch_users)).to(world.DEVICE)
        negitems = NegItems[idx, top1]
    return negitems

if __name__ == "__main__":
    method = UniformSample_DNS
    from register import dataset
    from utils import timer
    for i in range(1):
        with timer():
            # S = method(dataset, 1)
            S = UniformSample_original(dataset)
            print(len(S[S>= dataset.m_items]))
            S = torch.from_numpy(S).long()
            print(len(S[S >= dataset.m_items]))
        print(timer.get())
