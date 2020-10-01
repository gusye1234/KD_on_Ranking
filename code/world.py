'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
ROOT_PATH = '/Users/gus/Desktop/KD'
CODE_PATH = os.path.join(ROOT_PATH, 'code')
FILE_PATH = os.path.join(CODE_PATH, 'checkpoints')
BOARD_PATH = os.path.join(CODE_PATH, 'runs')
DATA_PATH = os.path.join(ROOT_PATH, 'data')
args = parse_args()
ARGS = args
EMBEDDING = args.embedding

config = {}
all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon']
all_models  = ['mf', 'lgn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['teacher_dim'] = args.teacher_dim
config['teacher_layer'] = args.teacher_layer
config['teacher_model'] = 'lgn'
DNS_K = args.dns_k
method = args.method
if method=='dns' and DNS_K == 1:
    method = 'original'

GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
SEED = args.seed

dataset = args.dataset
model_name = args.model
# if dataset not in all_dataset:
    # raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")




TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
ALLDATA = args.alldata
TESTDATA = args.testdata
ONE = args.one
if ONE and TESTDATA:
    raise TypeError("levave-one-out data shouldn't be trained only by test!!!")
T = args.T
beta = args.beta
p0 = args.p0
startepoch = args.startepoch
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
if ONE:
    dataset = dataset + "_one"
    topks = [25,50,100]
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)