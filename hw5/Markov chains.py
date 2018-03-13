import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

data_score = np.array(pd.read_csv('CFB2016_scores.csv',header = -1))
teamnames = np.array(open('TeamNames.txt','r').read().split('\n'))

M = np.zeros((760,760))
w = np.ones((1,760))/760.0
for items in data_score:
    if items[1] > items[3]:
        r = items[1] / double(items[1] + items[3])
        M[items[2] - 1, items[0] - 1] = M[items[2] - 1, items[0] - 1] + 1 + r
        M[items[0] - 1, items[0] - 1] = M[items[0] - 1, items[0] - 1] + 1 + r
        M[items[2] - 1, items[2] - 1] = M[items[2] - 1, items[2] - 1] + 1 - r
        M[items[0] - 1, items[2] - 1] = M[items[0] - 1, items[2] - 1] + 1 - r
    else:
        r = items[3] / double(items[1] + items[3])
        M[items[2] - 1, items[2] - 1] = M[items[2] - 1, items[2] - 1] + 1 + r
        M[items[0] - 1, items[2] - 1] = M[items[0] - 1, items[2] - 1] + 1 + r
        M[items[2] - 1, items[0] - 1] = M[items[2] - 1, items[0] - 1] + 1 - r
        M[items[0] - 1, items[0] - 1] = M[items[0] - 1, items[0] - 1] + 1 - r

M = M/np.sum(M,axis = 1).reshape((760,1))
w_r = np.empty((4,760))
u,v = np.linalg.eig(np.transpose(M))
wr = v[:,np.argsort(-u)[0]]
wr = wr/np.sum(wr)
obj = []
for i in xrange(10000):
    w = np.dot(w,M)
    if i == 9:
        print "w10:",w[:,np.argsort(-w)[0][:25]]
        print "team:",teamnames[np.argsort(-w)[0][:25]]
    if i == 99:
        print "w100:",w[:,np.argsort(-w)[0][:25]]
        print "team:", teamnames[np.argsort(-w)[0][:25]]
    if i == 999:
        print "w1000:",w[:,np.argsort(-w)[0][:25]]
        print "team:", teamnames[np.argsort(-w)[0][:25]]
    if i == 9999:
        print "w10000:",w[:,np.argsort(-w)[0][:25]]
        print "team:", teamnames[np.argsort(-w)[0][:25]]
    obj.append(np.sum(np.abs(w - wr)))
print "w_final:",abs(wr[np.argsort(-wr)[:25]])
print "team:", teamnames[np.argsort(-wr)[:25]]
plt.figure(1)
plt.plot(xrange(10000), obj)
plt.xlabel('iterations')
plt.ylabel('difference')
plt.show()