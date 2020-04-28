import os
import world
import dataloader
import model
import utils
import Procedure
from pprint import pprint

data_path = os.path.join(
                    world.DATA_PATH, 
                    world.dataset)

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path=data_path)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM(path=data_path)

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("Train Method:", world.method)
if world.method == 'dns':
    print(">>DNS K:", world.DNS_K)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}

TRAIN = {
    'original': Procedure.BPR_train_original,
    # 'dns': Procedure.BPR_train_DNS_neg
    'dns': Procedure.BPR_train_DNS_batch
}

DISTILL_TRAIN = {
    'experiment': Procedure.Distill_train
}