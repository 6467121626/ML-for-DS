import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



F = lambda x:-np.log(x+(10**-16))

data = open('nyt_data.txt','r').read().split('\n')
X = np.zeros((3012,8447))
W = np.random.uniform(low=1.0, high=2.0, size=(3012, 25))
H = np.random.uniform(low=1.0, high=2.0, size=(25, 8447))
for index_doc,doc in enumerate(data):
    for word in doc.split(','):
        index_word,count = word.split(':')
        X[int(index_word) - 1,index_doc] = count

obj = []
for i in xrange(100):
    H = H*np.dot(np.transpose(W), X/(np.dot(W,H) + (10**-16)))/(np.sum(W,axis = 0).reshape((25,1))+(10**-16))
    W = W*np.dot( X/(np.dot(W,H) + (10**-16)),np.transpose(H))/(np.sum(H,axis = 1).reshape((1,25))+(10**-16))
    o = np.sum(X*map(F,np.dot(W,H))+np.dot(W,H))
    obj.append(o)
    print i,":",o

W = W/np.sum(W,axis = 0).reshape((1,25))
plt.figure(1)
plt.plot(xrange(100), obj)
plt.xlabel('iterations')
plt.ylabel('objective')
plt.show()
voc = pd.read_csv('nyt_vocab.dat', sep=" ", header=None).values.flatten()
# print voc
for i in xrange(25):
    ind = np.argsort(-W[:,i])[:10]
    print W[ind,i]
    print voc[ind]