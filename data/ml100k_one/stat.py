from scipy.sparse import dok_matrix
import numpy as np
from tqdm import tqdm

n_user = 0
n_item = 0

plus = 0
with open('train.txt','r') as f:
    for line in f.readlines():
        li = line.split(' ')
        user = int(li[0])
        item = int(li[1])
        if user == 0:
            plus = 1
        if user > n_user:
            n_user = user
        if item > n_item:
            n_item = item
n_user += plus
n_item += plus
print("num_user : ", n_user)
print("num_user : ", n_item)

user_item_matrix = dok_matrix((n_user, n_item), dtype=np.int32)
item_user_matrix = dok_matrix((n_item, n_user), dtype=np.int32)
with open('train.txt','r') as f:
    for line in f.readlines():
        li = line.split(' ')
        user = int(li[0]) - (1-plus)
        item = int(li[1]) - (1-plus)
        user_item_matrix[user, item] = 1
        item_user_matrix[item, user] = 1
print(2)

user_rating = []
item_rating = []
for u in tqdm(range(n_user)):
    user_rating.append(len(user_item_matrix[u])+1)

for i in tqdm(range(n_item)):
    item_rating.append(len(item_user_matrix[i])+1)

print(min(user_rating), max(user_rating), np.mean(user_rating))
print(min(item_rating), max(item_rating), np.mean(item_rating))