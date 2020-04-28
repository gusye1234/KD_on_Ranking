import os
import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure

# ============================================================================
# ============================================================================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ============================================================================
# ============================================================================
# init model
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
procedure = register.TRAIN[world.method]
bpr = utils.BPRLoss(Recmodel, world.config)
# ============================================================================
# ============================================================================
file = utils.getFileName(world.model_name, world.dataset, world.config['latent_dim_rec'], layers=world.config['lightGCN_n_layers'])
weight_file = os.path.join(world.FILE_PATH, file)
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}") 
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
# ============================================================================
# ============================================================================
earlystop = utils.EarlyStop(patience=10, model=Recmodel, filename=weight_file)
Neg_k = 1
Recmodel = Recmodel.to(world.device)
# ============================================================================
# ============================================================================
# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter( os.path.join(world.BOARD_PATH,time.strftime("%m-%d-%Hh%Mm%Ss-") + world.method + str(world.DNS_K) + file + '_' + world.comment))
else:
    w = None
    world.cprint("not enable tensorflowboard")
    
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
            print(f"[TEST TIME] {time.time() - start}")
            if earlystop.step(epoch,results):
                print("trigger earlystop")
                print(f"best epoch:{earlystop.best_epoch}")
                print(f"best results:{earlystop.best_result}")
                break
finally:
    if world.tensorboard:
        w.close()