import os
import time
import world
import utils
import torch
import Procedure
import numpy as np
from pprint import pprint
from world import cprint
import sample
from tensorboardX import SummaryWriter

world.DISTILL = False
# ----------------------------------------------------------------------------
utils.set_seed(world.SEED)
print(f"[SEED:{world.SEED}]")
# ----------------------------------------------------------------------------
# init model
import register
from register import dataset

print(np.mean(dataset.popularity()[0]))

pop1 = np.loadtxt("popularity/popularity-gowa-teacher-mf.txt")
pop2 = np.loadtxt("popularity/popularity-gowa-RD.txt")
pop3 = np.loadtxt("popularity/popularity-gowa-student-mf.txt")

utils.draw(dataset, pop1, pop2, name2='RD-32')
# utils.powerlaw(pop1, pop2, pop3)
exit()
procedure = Procedure.Popularity_Bias
Recmodel = register.MODELS[world.model_name](world.config, dataset)
# ----------------------------------------------------------------------------
file = utils.getFileName(world.model_name,
                         world.dataset,
                         world.config['latent_dim_rec'],
                         layers=world.config['lightGCN_n_layers'])
file = 'teacher-' + file
weight_file = os.path.join(world.FILE_PATH, file)
print(f"Load {weight_file}")
utils.load(Recmodel, weight_file)
# ----------------------------------------------------------------------------
Recmodel = Recmodel.to(world.DEVICE)

pop = procedure(dataset, Recmodel)

np.savetxt(f"popularity-{world.dataset}-{world.comment}.txt", pop)