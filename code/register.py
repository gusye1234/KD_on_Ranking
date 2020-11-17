import os
import world
import dataloader
import model
import utils
import Procedure
import sample
from pprint import pprint

data_path = os.path.join(
                    world.DATA_PATH,
                    world.dataset)
if world.ONE:
    # data_path = data_path + "_one"
    print("{leave-one-out}:", data_path)

if world.dataset == 'lastfm':
    dataset = dataloader.LastFM(path=data_path)
else:
    if world.ONE:
        dataset = dataloader.LoaderOne(path=data_path)
    else:
        dataset = dataloader.Loader(path=data_path)

if world.DISTILL:
    print('===========DISTILL================')
    pprint(world.config)
    # print("beta:", world.beta)
    print("DNS K:", world.DNS_K)
    print("sample methods:", world.SAMPLE_METHOD)
    print("comment:", world.comment)
    print("tensorboard:", world.tensorboard)
    print("Test Topks:", world.topks)
    print('===========end===================')
else:
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
    'lgn': model.LightGCN,
    'leb': model.LightEmb
}

SAMPLER = {
    'combine' : sample.DistillSample,
    'RD'     : sample.RD,
    'CD'     : sample.CD
}

TRAIN = {
    'original': Procedure.BPR_train_original,
    'dns': Procedure.BPR_train_DNS_neg
    # 'dns': Procedure.BPR_train_DNS_batch
}

DISTILL_TRAIN = {
    'batch': Procedure.Distill_DNS_yield,
    'epoch': Procedure.Distill_DNS
}