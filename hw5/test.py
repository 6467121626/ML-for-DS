# coding: utf-8

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

data = pd.read_csv('nyt_data.txt', sep=" ", header=None)

data = data.values[:, :]
print data
N = 3012
M = 8447
K = 25

data1 = open('nyt_data.txt','r').read().split('\n')
X1 = np.zeros((3012,8447))
W = np.random.rand(3012,25)
H = np.random.rand(25,8447)
for index_doc,doc in enumerate(data1):
    for word in doc.split(','):
        index_word,count = word.split(':')
        X1[int(index_word) - 1,index_doc] = count


X = np.zeros([N, M])

for j in range(8447):
    list = data[j, 0].split(",")
    dic = {}
    for entry in list:
        key, val = entry.split(':')
        dic[key] = int(val)

    for k, v in dic.items():
        X[int(k)-1, j] += int(v)
    print j
print np.sum(X-X1),"1234"

W = np.random.uniform(low=1.0, high=2.0, size=(N, K))
H = np.random.uniform(low=1.0, high=2.0, size=(K, M))

O = []
for i in range(100):
    H = H * np.dot(np.transpose(W), (X / (np.dot(W, H)+(10**-16)))) / (np.sum(W, axis=0).reshape([K, 1]) + (10**-16))
    W = W * np.dot((X / (np.dot(W, H)+(10**-16))), np.transpose(H)) / (np.sum(H, axis=1).reshape([1, K]) + (10**-16))
    obj = np.sum(X * np.log(1/(np.dot(W, H)+(10**-16))) + np.dot(W, H))
    O.append(obj)
    print obj

voc = pd.read_csv('nyt_vocab.dat', header=None)
voc = voc.values[:, :]
voc = voc.tolist()

W_n = W / (np.sum(W, axis=0).reshape([1, K]) + (10**-16))
W_sort = np.zeros([10, 25])
W_idx = np.zeros([10, 25])
for i in range(25):
    temp = np.sort(-W_n[:, i])
    W_sort[:, i] = -temp[0:10]
    temp1 = np.argsort(-W_n[:, i])
    W_idx[:, i] = temp1[0:10]
    temp1 = temp1[0:10]
    print -temp[0:10]
    print [voc[j] for j in temp1]

plt.figure(1)
NN = range(100)
plt.plot(NN, O)
plt.xlabel('x')
plt.ylabel('y')
plt.title("HW5_1")
plt.savefig('HW5_1')
