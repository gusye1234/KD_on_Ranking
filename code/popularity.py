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
    name1 = "gowa-mf50"
    pop1 = np.loadtxt(f"stats/popularity-{name1}.txt")
    pop1_user = np.loadtxt(f"stats/popularity-{name1}-user.txt")

    name2 = 'gowa-mf200'
    pop2 = np.loadtxt(f"stats/popularity-{name2}.txt")
    pop2_user = np.loadtxt(f"stats/popularity-{name2}-user.txt")

    # utils.draw_longtail(dataset, pop1, pop2)
    utils.draw(dataset, pop1, pop2, name1, name2)
    # ----------------------------------------------------------------------------
    # name = "amaz-mf"
    # dims = [10, 50, 100, 150, 200]

    # names = [name + str(d) for d in dims]

    # pop_item = [
    #     np.loadtxt(f"stats/popularity-{n}.txt")
    #     for n in names
    # ]

    # pop_user = [
    #     np.loadtxt(f"stats/popularity-{n}-user.txt")
    #     for n in names
    # ]
    # for i in range(len(dims)):
    #     pop1 = pop_item[i]
    #     pop1_user = pop_user[i]
    #     n = names[i]
    #     # print(n)
    #     pprint(utils.popularity_ratio(pop1, pop1_user, dataset))
    #     print(',')

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
    pop, user_pop = procedure(dataset, Recmodel)

    np.savetxt(f"popularity-{world.dataset}-{world.comment}.txt", pop)
    np.savetxt(f"popularity-{world.dataset}-{world.comment}-user.txt", user_pop)