'''
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import os
import world
import torch
import random
import numpy as np
from time import time
from model import LightGCN
from torch import nn, optim
from torch import log, Tensor
from model import PairWiseModel
from dataloader import BasicDataset

# ============================================================================
# ============================================================================
# pair loss
class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)

    def stageOne(self,
                 users,
                 pos,
                 neg,
                 weights=None,
                 add_loss : torch.Tensor=None):
        # if world.CD == True:
        #     return self.cd_loss(users, pos, weights, add_loss)
        loss, reg_loss = self.model.bpr_loss(users, pos, neg, weights=weights)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss
        if add_loss is not None:
            assert add_loss.requires_grad == True
            # print(loss.item(), add_loss.item())
            loss = loss + add_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()



def getTestweight(users   : Tensor,
                  items   : Tensor,
                  dataset : BasicDataset):
    """
        designed only for levave-one-out data
    """
    users = users.cpu().numpy().astype('int')
    items = items.cpu().numpy().astype('int')
    testdict = dataset.testDict
    test_items = []
    for user in users:
        test_item = testdict[user][0]
        test_items.append(test_item)
    test_items = np.array(test_items).astype('int')
    index = (test_items == items)
    weights = np.ones_like(users)
    weights[index] = world.ARGS.testweight

    return Tensor(weights).to(world.DEVICE)

# ============================================================================
# ============================================================================
# utils
class EarlyStop:
    def __init__(self, patience, model, filename):
        self.patience = patience
        self.model = model
        self.filename = filename
        self.suffer = 0
        self.best = 0
        self.best_result = None
        self.best_epoch = 0
        self.mean = 0
        self.sofar = 1

    def step(self, epoch, performance):
        if performance['ndcg'][-1] < self.mean:
            self.suffer += 1
            if self.suffer >= self.patience:
                return True
            self.sofar += 1
            self.mean = self.mean*(self.sofar - 1)/self.sofar + performance['ndcg'][-1]/self.sofar
            print(f"no good so far {self.suffer}:{self.mean}")
        else:
            self.suffer = 0
            self.mean = performance['ndcg'][-1]
            self.sofar = 1
            self.best = performance['ndcg'][-1]
            self.best_result = performance
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), self.filename)
            return False

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName(model_name, dataset,rec_dim, layers=None):
    if model_name == 'mf':
        file = f"mf-{dataset}-{rec_dim}.pth.tar"
    elif model_name == 'lgn':
        assert layers is not None
        file = f"lgn-{dataset}-{layers}-{rec_dim}.pth.tar"
    return file

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def TO(*tensors, **kwargs):
    results = []
    for tensor in tensors:
        results.append(tensor.to(world.DEVICE))
    return results

def shapes(*tensors):
    shape = [tensor.size() for tensor in tensors]
    strs = [str(sh) for sh in shape]
    print(" : ".join(strs))


def getTeacherConfig(config : dict):
    teacher_dict = config.copy()
    teacher_dict['lightGCN_n_layers'] = teacher_dict['teacher_layer']
    teacher_dict['latent_dim_rec'] = teacher_dict['teacher_dim']
    return teacher_dict

def time2str(sam_time : list):
    sam_copy = ""
    for t in sam_time:
        sam_copy += '+' + f"{t:.2f}"
    return sam_copy[1:]

class Timer:
    """
    Time context manager for code block
    """
    from time import time
    def __init__(self, tape):
        self.tape = tape
    def __enter__(self):
        self.start = Timer.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tape.append(Timer.time()-self.start)


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE
            

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


def load(model, file):
    try:
        model.load_state_dict(torch.load(file))
    except RuntimeError:
        model.load_state_dict(torch.load(file, map_location=torch.device('cpu')))
    except FileNotFoundError:
        raise FileNotFoundError(f"{file} NOT exist!!!")

def combinations(start, end, com_num=2):
    """get all the combinations of [start, end]"""
    from itertools import combinations
    index = np.arange(start, end)
    return np.asanyarray(list(combinations(index, com_num)))


def display_top(snapshot, key_type='lineno', limit=3):
    import tracemalloc
    import linecache

    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB" %
              (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


# ============================================================================
# ============================================================================
# metrics
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}

def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def NDCGatK_r_ONE(r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    pred_data = r[:, :k]

    idcg = 1
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    ndcg = dcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def HRatK_ONE(r,k):
    pred = r[:, :k]
    return np.sum(pred)

def getLabel_ONE(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i][0]
        predictTopK = pred_data[i]
        pred = (predictTopK == groundTrue)
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')



# ====================end Metrics=============================
# =========================================================