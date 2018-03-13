import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from collections import defaultdict
from numpy.linalg import inv

def nomalize(arr):
    for i in xrange(54,57):
        arr[:, i] = 10*arr[:,i]/mean(arr[:, i])
    return arr

x_train = pd.read_csv('X_train.csv',header = -1)
y_train = pd.read_csv('y_train.csv',header = -1)
x_test = pd.read_csv('X_test.csv',header = -1)
y_test = pd.read_csv('y_test.csv',header = -1)
x_train = double(np.array(x_train))
y_train = double(np.array(y_train))
x_test = double(np.array(x_test))
y_test = double(np.array(y_test))


Sigmoid = lambda x:1-1/(1+exp(x))
y_train = 2*y_train - 1
y_test = 2*y_test - 1
x_train_logistic = np.ones((4508,58))
x_train_logistic[:,:57] = nomalize(x_train)
L = zeros(10000)
w = zeros(58)

for i in xrange(10000):
    eta = 1.0/(100000*np.sqrt(i+1))
    yxw = y_train.reshape(4508)*np.dot(x_train_logistic,w)
    sigm = np.array(map(Sigmoid,yxw))
    L[i] = sum(map(lambda x:np.log(x),sigm))
    deltaL = sum((1 - sigm.reshape(4508,1)) * y_train * x_train_logistic,axis = 0)
    w = w + eta*deltaL

print L
plt.plot(range(10000), L)
plt.show()