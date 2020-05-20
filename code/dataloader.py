"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

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
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.__trainsize = 0
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

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
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
        
        if world.ALLDATA:
            self._trainUser = self.trainUser
            self._trainItem = self.trainItem
            self.trainUser = np.concatenate([self.trainUser, self.testUser])
            self.trainItem = np.concatenate([self.trainItem, self.testItem])
            self.__trainsize += self.testDataSize
        elif world.TESTDATA:
            self.__trainsize = self.testDataSize
            self.trainUser = self.testUser
            self.trainItem  = self.testItem
        
        self.Graph = None
        print(f"({self.n_users} X {self.m_items})")
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.__n_users, self.__m_items))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self.__allPos = self.getUserPosItems(list(range(self.__n_users)))
        self.__testDict = self.build_test()
        if world.ALLDATA:
            self.UserItemNet = csr_matrix((np.ones(len(self._trainUser)), (self._trainUser, self._trainItem)),
                                      shape=(self.__n_users, self.__m_items))
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

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
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
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def getP(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                a = 0.05
                P = sp.load_npz(self.path + '/P.npz')
                print("using P====", "alpha is", a)
                print("successfully loaded...")
                norm_adj = P
            except :
                print("generating P")
                s = time()
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("load pre")
                item_model = sp.load_npz(self.path + '/item_model.npz')
                item_model = item_model.tolil()
                print("item model", item_model.shape)
                
                M = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                M = M.tolil()
                I = sp.eye(self.n_users).tolil()
                M[:self.n_users, :self.n_users] = I
                M[self.n_users:, self.n_users:] = item_model
                M = M.tocsr()
                
                norm_adj = (1-a)*pre_adj_mat + a*M
                
                sp.save_npz(self.path + '/P.npz', norm_adj)
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

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
        test_file = path + '/test.txt'        
        trainUser, trainItem = [], []
        testUser, testItem = [], []
        with open(train_file) as f:
            for line in f.readlines():
                user, item, _ = line.strip().split()
                trainUser.append(int(user))
                trainItem.append(int(item))
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
        self.testUser = np.array(testUser) - min_index
        self.testItem = np.array(testItem) - min_index
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
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + len(testUser)) / self.n_users / self.m_items}")
        
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_users, self.m_items))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self.__allPos = self.getUserPosItems(list(range(self.__n_users)))
        self.__testDict = self.build_test()
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
    def allPos(self):
        return self.__allPos

# ----------------------------------------------------------------------------
# this dataset is for debugging
class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../data/lastfm"):
        import pandas as pd
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # (users,users)
        if world.TESTDATA:
            self.trainUser = self.testUser
            self.trainItem  = self.testItem
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        
        # pre-calculate
        self.__allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self.__allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.build_test()

    @property
    def n_users(self):
        return 1892
    
    @property
    def m_items(self):
        return 4489
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self.__allPos

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
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
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            