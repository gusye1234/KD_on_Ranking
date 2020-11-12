from cppimport import imp, imp_from_filepath
import numpy as np
from pprint import pprint
import os
import sys
import torch
sys.path.append("/Users/gus/Desktop/KD/code/sources")
os.environ['CC'] = '/usr/local/opt/llvm/bin/clang++'

np.random.seed(2020)

print(os.environ['CC'])

b = [np.array([1, 2, 3]).astype('int'), np.array([11, 12, 13]).astype('int')]

b = np.random.randint(14009, size=(13149, 50)).astype('int')



sample = imp_from_filepath("/Users/gus/Desktop/KD/code/sources/sampling.cpp")
# sample.seed(2020)

data = sample.sample_negative(13149, 14009, 574956, b, 5)
# print(data.shape)
print(data[:3])

# users = np.random.randint(10000, size=(1000,)).astype("int")
# data = sample.sample_negative_ByUser(users, 14009, b, 10)
# print(data.shape)
# print(data[:5])
# print(users[:5])
# pprint(data[:5])
# print(torch.from_numpy(data).long()[:5])
# print(data[41:43 + 5])
