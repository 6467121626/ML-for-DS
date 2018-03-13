import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from collections import defaultdict
from numpy.linalg import inv
x_train = pd.read_csv('X_train.csv',header = -1)
y_train = pd.read_csv('y_train.csv',header = -1)
x_test = pd.read_csv('X_test.csv',header = -1)
y_test = pd.read_csv('y_test.csv',header = -1)
x_train = double(np.array(x_train))
y_train = double(np.array(y_train))
x_test = double(np.array(x_test))
y_test = double(np.array(y_test))
# naive Bayes classifier
p = double(sum(y_train))/y_train.shape[0]
#calculate the parameter of first 54 bernouli values
part1 = x_train[:,:54]
L = zeros(54)
K = zeros(54)
M = zeros(54)
N = zeros(54)
for i in xrange(part1.shape[0]):
    for j in xrange(part1.shape[1]):
        if (part1[i][j] == 1) & (y_train[i] == 1):
            L[j] = L[j] + 1
        if (part1[i][j] == 0) & (y_train[i] == 1):
            K[j] = K[j] + 1
        if (part1[i][j] == 1) & (y_train[i] == 0):
            M[j] = M[j] + 1
        if (part1[i][j] == 0) & (y_train[i] == 0):
            N[j] = N[j] + 1
theta11 = L/(L+K)
theta10 = M/(N+M)
#calculate the parameter of 3 Pareto distribution values
part2 = np.array(map(lambda x: np.log(x), x_train[:,54:]))
lnsum1 = zeros(3)
lnsum0 = zeros(3)

for i in xrange(part2.shape[0]):
    for j in xrange(part2.shape[1]):
         if y_train[i] == 1:
             lnsum1[j] = lnsum1[j] + part2[i][j]
         if y_train[i] == 0:
             lnsum0[j] = lnsum0[j] + part2[i][j]

theta21 = sum(y_train)/lnsum1
theta20 = (part2.shape[0]-sum(y_train))/lnsum0


# calsulate the result based on the parameters we get before
re1 = double(np.empty_like(x_test))
re0 = double(np.empty_like(x_test))
for i in xrange(re1.shape[0]):
    for j in xrange(54):
        if  x_test[i][j] == 1:
            re1[i][j] = theta11[j]
            re0[i][j] = theta10[j]
        if  x_test[i][j] == 0:
            re1[i][j] = 1 - theta11[j]
            re0[i][j] = 1 - theta10[j]
for i in xrange(re1.shape[0]):
    for j in xrange(54,57):
        re1[i][j] = theta21[j - 54] * (x_test[i][j] ** (-theta21[j - 54] - 1))
        re0[i][j] = theta20[j - 54] * (x_test[i][j] ** (-theta20[j - 54] - 1))
p1 = zeros(x_test.shape[0])
p0 = zeros(x_test.shape[0])
for i in xrange(x_test.shape[0]):
    p1[i] = reduce(lambda x,y:x*y,re1[i,:])*p
    p0[i] = reduce(lambda x,y:x*y,re0[i,:])*(1-p)

y_predict = zeros(x_test.shape[0])
for i in xrange(x_test.shape[0]):
    if p1[i] > p0[i]:
        y_predict[i] = 1
# (y; y0)-th cell of the table, where y and y0 can be either 0 or 1.p means positive n means negative
pp = sum(y_predict*y_test.reshape(93))
nn = sum((1-y_predict)*(1-y_test.reshape(93)))
pn = sum(y_predict*(1-y_test.reshape(93)))
np = sum((1-y_predict)*y_test.reshape(93))
accuracy = (pp+nn)/93
print accuracy,pp,np,pn,np
plt.stem(range(54), theta11,"-.",label='dotted line:spam')
plt.stem(range(54), theta10," ",label='no line:non-spam')
plt.title('stem plot of 54 Bernoulli parameters')
plt.legend(loc=2)
plt.show()