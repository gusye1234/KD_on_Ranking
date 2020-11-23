"""
@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
import sys
import torch
import world
import numpy as np
import scipy.sparse as sp
from time import time
from world import cprint
from os.path import join
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError
# ----------------------------------------------------------------------------
class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self,config = world.config,path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.__n_users = 0
        self.__m_items = 0
        train_file = path + '/train.txt'
        valid_file = path + '/valid.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        validUniqueUsers, validItem, validUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.__trainsize = 0
        self.validDataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.__m_items = max(self.__m_items, max(items))
                    self.__n_users = max(self.__n_users, uid)
                    self.__trainsize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(valid_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    validUniqueUsers.append(uid)
                    validUser.extend([uid] * len(items))
                    validItem.extend(items)
                    self.__m_items = max(self.__m_items, max(items))
                    self.__n_users = max(self.__n_users, uid)
                    self.validDataSize += len(items)
        self.validUniqueUsers = np.array(validUniqueUsers)
        self.validUser = np.array(validUser)
        self.validItem = np.array(validItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    try:
                        items = [int(i) for i in l[1:]]
                    except:
                        print("user data error", l)
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.__m_items = max(self.__m_items, max(items))
                    self.__n_users = max(self.__n_users, uid)
                    self.testDataSize += len(items)
        self.__m_items += 1
        self.__n_users += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        # if world.ALLDATA:
        #     self._trainUser = self.trainUser
        #     self._trainItem = self.trainItem
        #     self.trainUser = np.concatenate([self.trainUser, self.testUser])
        #     self.trainItem = np.concatenate([self.trainItem, self.testItem])
        #     self.__trainsize += self.testDataSize
        # elif world.TESTDATA:
        #     self.__trainsize = self.testDataSize
        #     self.trainUser = self.testUser
        #     self.trainItem  = self.testItem

        self.Graph = None
        print(f"({self.n_users} X {self.m_items})")
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.validDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{world.dataset} Sparsity : {(self.trainDataSize + self.validDataSize + self.testDataSize) / self.n_users / self.m_items}"
        )

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.__n_users, self.__m_items), dtype='int')
        # pre-calculate
        self.__allPos = self.getUserPosItems(list(range(self.__n_users)))
        self.__testDict = self.build_dict(self.testUser, self.testItem)
        self.__validDict = self.build_dict(self.validUser, self.validItem)
        if world.ALLDATA:
            self.UserItemNet = csr_matrix((np.ones(len(self._trainUser)), (self._trainUser, self._trainItem)),
                                      shape=(self.__n_users, self.__m_items), dtype='int')
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.__n_users

    @property
    def m_items(self):
        return self.__m_items

    @property
    def trainDataSize(self):
        return self.__trainsize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self.__allPos

    def popularity(self):
        popularity = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        sorted_index = np.argsort(-popularity)
        return popularity, sorted_index

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.DEVICE))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                if world.ALLDATA:
                    print(f"[all data]{self.path}")
                    pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_all.npz')
                else:
                    pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')

                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                rowsum[rowsum == 0.] = 1.
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                if world.ALLDATA:
                    sp.save_npz(self.path + '/s_pre_adj_mat_all.npz', norm_adj)
                else:
                    sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.DEVICE)
        return self.Graph

    def build_dict(self, users, items):
        data = {}
        for i, item in enumerate(items):
            user = users[i]
            if data.get(user):
                data[user].append(item)
            else:
                data[user] = [item]
        return data


    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict
# ----------------------------------------------------------------------------
class LoaderOne(Loader):
    def __init__(self,
                 config=world.config,
                 path="../data/gowalla_one"):
        cprint(f'loading [{path}]')
        self.path = path
        self.split = False
        self.__n_users = 0
        self.__m_items = 0
        train_file = path + '/train.txt'
        valid_file = path + '/valid.txt'
        test_file = path + '/test.txt'
        trainUser, trainItem = [], []
        validUser, validItem = [], []
        testUser, testItem = [], []
        with open(train_file) as f:
            for line in f.readlines():
                user, item, _ = line.strip().split()
                trainUser.append(int(user))
                trainItem.append(int(item))
        with open(valid_file) as f:
            for line in f.readlines():
                user, item, _ = line.strip().split()
                validUser.append(int(user))
                validItem.append(int(item))
        with open(test_file) as f:
            for line in f.readlines():
                user, item, _ = line.strip().split()
                testUser.append(int(user))
                testItem.append(int(item))
        self.__n_users = len(testUser)
        self.__m_items = max(max(trainItem), max(testItem))
        self.__trainsize = len(trainUser)
        min_index = np.min(trainUser)
        self.trainUser = np.array(trainUser) - min_index
        self.trainItem = np.array(trainItem) - min_index
        self.validUser = np.array(validUser) - min_index
        self.validItem = np.array(validItem) - min_index
        self.testUser = np.array(testUser) - min_index
        self.testItem = np.array(testItem) - min_index
        self.__m_items += 1 - min_index
        assert len(testUser) == (max(trainUser) + 1 - min_index)
        if world.ALLDATA:
            self._trainUser = self.trainUser
            self._trainItem = self.trainItem
            self.trainUser = np.concatenate([self.trainUser, self.testUser])
            self.trainItem = np.concatenate([self.trainItem, self.testItem])
            self.__trainsize += len(testUser)
        elif world.TESTDATA:
            self.__trainsize = len(testUser)
            self.trainUser = self.testUser
            self.trainItem = self.testItem

        self.Graph = None
        print(f"({self.n_users} X {self.m_items})")
        print(f"{self.trainDataSize} interactions for training")
        print(f"{len(testUser)} interactions for testing")
        print(f"{len(validUser)} interactions for validating")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + len(validUser) + len(testUser)) / self.n_users / self.m_items}")

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_users, self.m_items))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self.__allPos = self.getUserPosItems(list(range(self.__n_users)))
        self.__testDict = self.build_dict(self.testUser, self.testItem)
        self.__validDict = self.build_dict(self.validUser, self.validItem)
        if world.ALLDATA:
            self.UserItemNet = csr_matrix((np.ones(len(self._trainUser)), (self._trainUser, self._trainItem)),
                                      shape=(self.n_users, self.m_items))
        print(f"{world.dataset} is ready to go")
    @property
    def n_users(self):
        return self.__n_users

    @property
    def m_items(self):
        return self.__m_items

    @property
    def trainDataSize(self):
        return self.__trainsize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    @property
    def allPos(self):
        return self.__allPos

# ----------------------------------------------------------------------------
# this dataset is for debugging