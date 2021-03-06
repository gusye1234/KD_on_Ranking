'''
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=2,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=1000,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='gowa',
                        help="available datasets: [gowa yelp amzo]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[10,25,50]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    parser.add_argument('--method', type=str, default='original', help='train process [original, dns]')
    parser.add_argument('--dns_k', type=int, default=8, help='The polynomial degree for DNS(Dynamic Negative Sampling)')
    parser.add_argument('--alldata', type=int, default=0, help='include test set to train')
    parser.add_argument('--testdata', type=int, default=0, help='only include test set to train')
    parser.add_argument('--testweight', type=float, default=1)
    parser.add_argument('--teacher_dim', type=int, default=50, help='teacher\'s dimension')
    parser.add_argument('--teacher_layer', type=int, default=2, help='teacher\'s layer')
    parser.add_argument('--startepoch', type=int, default=1, help='The epoch to start distillation')
    parser.add_argument('--T', type=float, default=1.0, help='The temperature for teacher distribution')
    parser.add_argument('--beta', type=float, default=1e-4, help='The beta')
    parser.add_argument('--p0', type=float, default=1.0, help='The p0')
    parser.add_argument('--one', type=int, default=0, help='leave one out')
    parser.add_argument('--embedding', type=int, default=0, help='enable embedding distillation')
    parser.add_argument('--sampler', type=str, default='combine')
    return parser.parse_args()