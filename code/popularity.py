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

draw = True

if draw:
    pop1 = np.loadtxt("popularity/popularity-gowa-student-mf.txt")
    pop1 = pop1/np.max(pop1)
    pop2 = np.loadtxt("popularity/popularity-gowa-RD.txt")
    pop2 = pop2/np.max(pop2)

    data_pop, sorted_index = dataset.popularity()
    data_pop = data_pop.astype("float")/np.max(data_pop)
    print(np.sum(data_pop), np.sum(pop1), np.sum(pop2))

    utils.draw(dataset, data_pop,pop1, pop2)
    # utils.powerlaw(pop1, pop2, pop3)
    # ----------------------------------------------------------------------------
else:
    procedure = Procedure.Popularity_Bias
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
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
    test_results = Procedure.Test(dataset, Recmodel, 0, valid=False)
    pprint(test_results)
    pop = procedure(dataset, Recmodel)

    np.savetxt(f"popularity-{world.dataset}-{world.comment}.txt", pop)