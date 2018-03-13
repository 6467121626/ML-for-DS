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
y_train = 2*y_train - 1
y_test = 2*y_test - 1
x_train_logistic = np.ones((4508,58))
x_train_logistic[:,:57] = x_train
Sigmoid = lambda x:1-1/(1+exp(x))
L2 = zeros(100)
w = zeros(58)
for i in xrange(100):
    eta = 1.0/(np.sqrt(i+1))
    yxw = y_train.reshape(4508) * np.dot(x_train_logistic, w)
    sigm = np.array(map(Sigmoid, yxw))
    L2[i] = sum(map(lambda x: np.log(x), sigm))
    deltaL = sum((1 - sigm.reshape(4508, 1)) * y_train * x_train_logistic, axis=0)
    delta2L = np.dot(sigm*(1-sigm)*np.transpose(x_train_logistic),x_train_logistic)
    w = w + eta*np.dot(inv(delta2L),deltaL)
plt.plot(range(100), L2)
plt.show()

x_test_logistic = np.ones((93,58))
x_test_logistic[:,:57] = x_test
def ind(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    else:
        return 0
prediction2 = np.array(map(ind,np.dot(x_test_logistic,w)))
accuracy = sum(1 +  prediction2 * y_test.reshape(93))/(2*x_test.shape[0])
print accuracy
