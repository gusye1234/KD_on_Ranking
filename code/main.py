'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Main program
'''
import os
import time
import world
import utils
import torch
import Procedure
import numpy as np
from pprint import pprint
from world import cprint
from tensorboardX import SummaryWriter

# ----------------------------------------------------------------------------
utils.set_seed(world.SEED)
print(f"[SEED:{world.SEED}]")
# ----------------------------------------------------------------------------
# init model
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
procedure = register.TRAIN[world.method]
bpr = utils.BPRLoss(Recmodel, world.config)
# ----------------------------------------------------------------------------
file = utils.getFileName(world.model_name, world.dataset, world.config['latent_dim_rec'], layers=world.config['lightGCN_n_layers'])
weight_file = os.path.join(world.FILE_PATH, file)
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}") 
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
# ----------------------------------------------------------------------------
earlystop = utils.EarlyStop(patience=30, model=Recmodel, filename=weight_file)
Neg_k = 1
Recmodel = Recmodel.to(world.DEVICE)
# ----------------------------------------------------------------------------
# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
        os.path.join(
            world.BOARD_PATH,time.strftime("%m-%d-%Hh-%Mm-") + f"{world.method}-{str(world.DNS_K)}-{file.split('.')[0]}-{world.comment}"
            )
        )
else:
    w = None
    world.cprint("not enable tensorflowboard")
# ----------------------------------------------------------------------------
# start training
try:
    for epoch in range(world.TRAIN_epochs):
        print('======================')
        print(f'EPOCH[{epoch}/{world.TRAIN_epochs}]')
        start = time.time()
        output_information = procedure(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        
        print(f'[saved][{output_information}]')
        print(f"[TOTAL TIME] {time.time() - start}")
        if epoch %3 == 0:
            start = time.time()
            cprint("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            pprint(results)
            print(f"[TEST TIME] {time.time() - start}")
            if earlystop.step(epoch,results):
                print("trigger earlystop")
                print(f"best epoch:{earlystop.best_epoch}")
                print(f"best results:{earlystop.best_result}")
                break
finally:
    if world.tensorboard:
        w.close()