'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Distill model
'''
import os
import time
import world
import utils
import torch
import Procedure
import numpy as np
from world import cprint
from pprint import pprint
from tensorboardX import SummaryWriter
from sample import DistillSample, LogitsSample

# ----------------------------------------------------------------------------
# global
world.DISTILL = True
# ----------------------------------------------------------------------------
# set seed
utils.set_seed(world.SEED)
print(f"[SEED:{world.SEED}]")
# ----------------------------------------------------------------------------
# init model
import register
from register import dataset

# ----------------------------------------------------------------------------
# loading teacher
teacher_file = utils.getFileName(world.model_name,
                                 world.dataset,
                                 world.config['teacher_dim'],
                                 layers=world.config['teacher_layer'])
teacher_weight_file = os.path.join(world.FILE_PATH, teacher_file)
print('-------------------------')
world.cprint("loaded teacher weights from")
print(teacher_weight_file)
print('-------------------------')
teacher_config = utils.getTeacherConfig(world.config)
world.cprint('teacher')
teacher_model = register.MODELS[world.model_name](teacher_config,
                                                  dataset,
                                                  fix=True)
teacher_model.eval()
utils.load(teacher_model, teacher_weight_file)




world.cprint('student')
if world.EMBEDDING:
    student_model = register.MODELS['leb'](world.config, dataset, teacher_model)
    print(student_model)
else:
    student_model = register.MODELS[world.model_name](world.config, dataset)

procedure = register.DISTILL_TRAIN['experiment']
# procedure = register.DISTILL_TRAIN['logits']
bpr = utils.BPRLoss(student_model, world.config)
sampler = DistillSample(dataset,
                        student_model,
                        teacher_model,
                        world.DNS_K)
# sampler = LogitsSample(dataset, student_model, teacher_model, world.DNS_K)

# ----------------------------------------------------------------------------
# get names
file = utils.getFileName(world.model_name, world.dataset, world.config['latent_dim_rec'], layers=world.config['lightGCN_n_layers'])
weight_file = os.path.join(world.FILE_PATH, file)
print('-------------------------')
print(f"load and save student to {weight_file}")
if world.LOAD:
    utils.load(student_model, weight_file)
# ----------------------------------------------------------------------------
# migrate and stuffs
earlystop = utils.EarlyStop(patience=60, model=student_model, filename=weight_file)
Neg_k = 1
student_model = student_model.to(world.DEVICE)
teacher_model = teacher_model.to(world.DEVICE)
# ----------------------------------------------------------------------------
# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
        os.path.join(
            world.BOARD_PATH,time.strftime("%m-%d-%Hh-%Mm-") + f"{world.method}-{str(world.DNS_K)}-{file.split('.')[0]}-{world.comment}-DISTILL"
            )
        )
else:
    w = None
    world.cprint("not enable tensorflowboard")
# ----------------------------------------------------------------------------
# test teacher
cprint("[TEST Teacher]")
results = Procedure.Test(dataset, teacher_model, 0, None, world.config['multicore'])
pprint(results)
# ----------------------------------------------------------------------------
# start training
try:
    for epoch in range(world.TRAIN_epochs):

        start = time.time()
        output_information = procedure(dataset, student_model, sampler, bpr, epoch, w=w)

        print(
            f'EPOCH[{epoch}/{world.TRAIN_epochs}][{time.time() - start:.2f}] - {output_information}'
        )
        if epoch %3 == 0:
            start = time.time()
            cprint("[TEST]", ends=':')
            results = Procedure.Test(dataset, student_model, epoch, w, world.config['multicore'])
            pprint(results)
            # print(f"[TEST TIME] {time.time() - start}")
            if earlystop.step(epoch,results):
                print("trigger earlystop")
                print(f"best epoch:{earlystop.best_epoch}")
                print(f"best results:{earlystop.best_result}")
                break
finally:
    if world.tensorboard:
        w.close()