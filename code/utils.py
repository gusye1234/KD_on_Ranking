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
from dataloader import BasicDataset, Loader

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
        self.best_model = model.state_dict()
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
            print(
                f"******************Suffer {performance['ndcg'][-1]}:{self.mean}"
            )
        else:
            self.suffer = 0
            self.mean = performance['ndcg'][-1]
            self.sofar = 1
            self.best = performance['ndcg'][-1]
            self.best_result = performance
            self.best_epoch = epoch
            self.best_model = self.model.state_dict()
            torch.save(self.best_model, self.filename)
            return False

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName(model_name, dataset,rec_dim, layers=None, dns_k=None):
    if model_name == 'mf':
        if dns_k is not None:
            file = f"mf-{dataset}-{rec_dim}-{dns_k}.pth.tar"
        else:
            file = f"mf-{dataset}-{rec_dim}.pth.tar"
    elif model_name == 'lgn':
        assert layers is not None
        if dns_k is not None:
            file = f"lgn-{dataset}-{layers}-{rec_dim}-{dns_k}.pth.tar"
        else:
            file = f"lgn-{dataset}-{layers}-{rec_dim}.pth.tar"
    return file

def getLogFile():
    model = world.model_method
    comment = world.comment
    dataset = world.dataset
    return f"{dataset}-{model}-{comment}.txt"

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

# Draw and Count<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def draw(dataset, pop_rate, pop1, pop2,):
    import matplotlib.pyplot as plt
    import powerlaw
    dataset : Loader
    # pop_item, index = dataset.popularity()
    x = pop_rate

    pop1_mask = (pop1 > x)
    pop2_mask = (pop2 > x)

    plt.scatter(x[~pop1_mask], pop1[~pop1_mask], c='springgreen', linewidth=0, s=10, alpha=1, label='student')
    plt.scatter(x[~pop2_mask], pop2[~pop2_mask], c='blue', s=10, linewidth=0, alpha=0.3,label="After distillation")

    plt.scatter(x[pop2_mask],pop2[pop2_mask], c='blue', s=30, linewidth=0, alpha=0.8, label="After distillation")
    plt.scatter(x[pop1_mask], pop1[pop1_mask], c='springgreen', linewidth=0, s=30, alpha=1, label='student')

    plt.plot(x, x, linewidth=8,label="dataset")
    plt.xlabel("Dataset popularity rate")
    plt.ylabel("Model popularity rate")
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 1, 4, 2, 3]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    plt.legend(handles, labels)
    plt.title("Gowalla")
    plt.show()

def draw_help_log(x, num):
    x_index = np.log2(1 + x * num)
    x_index = x_index / x_index.max()
    return x_index

def draw_log(
    dataset,
    pop_rate,
    pop1,
    pop2,
):
    import matplotlib.pyplot as plt
    import powerlaw
    dataset: Loader
    # pop_item, index = dataset.popularity()
    x = pop_rate
    x_index = draw_help_log(x, 100)
    pop1 = draw_help_log(pop1, 100)
    pop2 = draw_help_log(pop2, 100)
    # plt.plot(x, pop1, c='springgreen', linewidth=15, alpha=0.8, label=name1)
    plt.scatter(x_index,
                pop1,
                c='green',
                linewidth=0,
                s=30,
                alpha=0.5,
                label='student')
    # plt.plot(x, pop1, c='darkgreen', linewidth=3, alpha=0.3, label=name2)
    plt.scatter(x_index,
                pop2,
                c='blue',
                s=30,
                linewidth=0,
                alpha=0.5,
                label="After distillation")

    plt.plot(np.sort(x_index), np.sort(x_index), label="dataset")
    plt.xlabel("Dataset popularity rate")
    plt.ylabel("Model popularity rate")
    plt.legend()
    plt.title("Gowalla")
    plt.show()


def powerlaw(pop1, pop2, pop3):
    import matplotlib.pyplot as plt
    import powerlaw
    fit1 = powerlaw.Fit(pop1)
    fit2 = powerlaw.Fit(pop2)
    fit3 = powerlaw.Fit(pop3)
    fig1 = fit1.plot_pdf(color='b', label="Teacher")
    fit1.power_law.plot_pdf(color='b', linestyle="--",ax=fig1)

    # fig1 = fit1.plot_ccdf(color='b', label="Teacher")
    # fit1.power_law.plot_ccdf(color='b', linestyle="--", ax=fig1)

    fit2.plot_pdf(color='y', ax=fig1, label="RD-32")
    fit2.power_law.plot_pdf(color='y', linestyle="--",ax=fig1)
    fit3.plot_pdf(color='r', ax=fig1, label="Student")
    fit3.power_law.plot_pdf(color='r', linestyle="--",ax=fig1)

    # fit2.plot_ccdf(color='y', ax=fig1, label="RD-32")
    # fit2.power_law.plot_ccdf(color='y', linestyle="--", ax=fig1)
    # fit3.plot_ccdf(color='r', ax=fig1, label="Student")
    # fit3.power_law.plot_ccdf(color='r', linestyle="--", ax=fig1)

    plt.xlabel("log(x)")
    plt.ylabel("log(#popularity)")
    plt.title("Probability Density Function")
    # plt.title("Complementary Cumulative Distribution Function")
    plt.legend()
    plt.show()

def map_item_three(pop_item):
    """mapping item into short-head(0.2), long-tail(0.6), distant-tail(0.2)

    Args:
        pop_item ([type]): [description]
        
    Return:
        list[ndarray...]: short-head, long-tail, distant-tail
    """
    from math import floor, ceil
    index = np.argsort(pop_item)[::-1]
    num_item = len(index)
    return (index[:floor(num_item * 0.2)],
            index[ceil(num_item * 0.2):floor(num_item * 0.8)],
            index[ceil(num_item * 0.8):])

def APT(pop_user, mappings):
    """calculate the APT metrics for different sets

    Args:
        pop_user (list | ndarray): the recommend list or history of users
        mappings (list | tuple): (short-head, long-tail, distant-tail)

    Returns:
        list: APTs for different mappings
    """
    total_set = len(mappings)
    total_user = len(pop_user)
    apts = []
    for mapping in mappings:
        apt = 0.
        for user_item in pop_user:
            count = list(map(lambda x: x in mapping, user_item))
            apt = np.mean(count)
        apt = apt/total_user
        apts.append(apt)
    return apts

def popularity_ratio(pop_model : np.ndarray,
                     pop_model_user : np.ndarray,
                     dataset : Loader):
    """calculate the degree of the "long-tailness" for a distribution

    Args:
        pop_model (ndarray): the freq of items recommended by model
        pop_model_user (ndarray): (user X topk) the recommend list of users
        dataset (dataloader.Loader): the freq of items in dataset

    Returns:
        dict: {"I_ratio""float, "I_KL":float, "I_gini":float, "APT":[], "I_bin": float}
    """
    pop_dataset, _ = dataset.popularity()
    
    assert len(pop_model) == len(pop_dataset)
    num_item = len(pop_dataset)
    num_interaction = pop_model.sum()
    metrics = {}

    metrics['I_ratio'] = pop_model.max() / pop.min()

    prop_model = pop_model / num_interaction
    prop_uniform = 1./num_item
    metrics['I_KL']= np.sum(prop_model*np.log(prop_model/prop_uniform))

    metrics['I_gini'] = 0.

    # metrics['APT'] = APT(pop_model_user,)

# Dataset spliting (only used once for generation)<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

def _loo_split_dataset(train_f, test_f):
    train = {}
    test = {}
    with open(train_f, 'r') as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip('\n').split()
                item = [int(i) for i in line[1:]]
                user = int(line[0])
                train[user] = item
    with open(test_f, 'r') as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip('\n').split()
                item = [int(i) for i in line[1:]]
                user = int(line[0])
                test[user] = item
    for key in list(test):
        train[key] = train[key] + test[key]

    all_data = train
    train_seq = []
    valid_seq = []
    test_seq = []
    users_list = sorted(list(train))
    for user in users_list:
        user_item = all_data[user]
        assert len(user_item)
        for t_item in user_item[:-2]:
            train_seq.append((user, t_item, 1))
        valid_seq.append((user, user_item[-2], 1))
        test_seq.append((user, user_item[-1], 1))
    train_file = "bptrain.txt"
    valid_file = "bpvalid.txt"
    test_file = "bytest.txt"
    with open(train_file, 'w') as f:
        for t_data in train_seq:
            f.write(f"{t_data[0]} {t_data[1]} {1}\n")
    with open(valid_file, 'w') as f:
        for t_data in valid_seq:
            f.write(f"{t_data[0]} {t_data[1]} {1}\n")
    with open(test_file, 'w') as f:
        for t_data in test_seq:
            f.write(f"{t_data[0]} {t_data[1]} {1}\n")

def _split_dataset(train_f, ratio=.1):
    train = {}
    with open(train_f, 'r') as f:
        for line in f.readlines():
            if len(line) > 0:
                line = line.strip('\n').split()
                item = [int(i) for i in line[1:]]
                user = int(line[0])
                train[user] = item
    valid = {}
    for user in list(train):
        items = np.array(train[user]).astype('int')
        valid_num = round(ratio * len(items))
        if valid_num > 0:
            index = np.random.choice(
                np.arange(len(items)), size=(valid_num, ), replace=False
            )
            train_index = np.ones_like(items).astype('bool')
            train_index[index] = False
            train_items = items[train_index]
            valid_items = items[~train_index]
            train[user] = train_items
            valid[user] = valid_items
        else:
            print(f"{user} has no items in VALID")

    train_file = "bptrain.txt"
    valid_file = "bpvalid.txt"
    with open(train_file, 'w') as f:
        users = sorted(list(train))
        for user in users:
            f.write(f"{user}")
            for u_item in train[user]:
                f.write(f" {int(u_item)}")
            f.write('\n')
    with open(valid_file, 'w') as f:
        users = sorted(list(valid))
        for user in users:
            f.write(f"{user}")
            for u_item in valid[user]:
                f.write(f" {int(u_item)}")
            f.write('\n')


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