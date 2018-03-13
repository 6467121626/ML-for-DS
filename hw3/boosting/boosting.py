import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

x_train = pd.read_csv('X_train.csv',header = -1)
y_train = pd.read_csv('y_train.csv',header = -1)
x_test = pd.read_csv('X_test.csv',header = -1)
y_test = pd.read_csv('y_test.csv',header = -1)

x_train = double(np.array(x_train))
y_train = double(np.array(y_train))
x_test = double(np.array(x_test))
y_test = double(np.array(y_test))
x_trainc = np.ones((1036,6))
x_testc = np.ones((1000,6))
x_trainc[:,1:] = x_train
x_testc[:,1:] = x_test

rounds = 1500
weight = np.ones((1036))/1036.0
alpha = np.ones(rounds)
w = {}
trainerror = np.empty(rounds)
testerror = np.empty(rounds)
error = np.empty(rounds)
traerrup = np.empty(rounds)
# hist = np.zeros(1036)
index = np.zeros(rounds*1036)
for T in xrange(rounds):
    index[T*1036:(T+1)*1036] = np.random.choice(1036,1036,p=weight)
    # h, bin_edges = np.histogram(index, bins=range(1037))
    # hist = np.add(hist,h)
    x_trainb = x_trainc[np.int32(index[T*1036:(T+1)*1036])]
    y_trainb = y_train[np.int32(index[T*1036:(T+1)*1036])]
    w[T] = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_trainb),x_trainb)),np.transpose(x_trainb)),y_trainb)
    prediction = np.sign(np.dot(x_trainc,w[T]))
    error[T] = np.dot(weight,((1-y_train*prediction)/2))
    alpha[T] = np.log((1 - error[T])/error[T])/2
    weightpara = np.array(map(lambda x:exp(x),-alpha[T]*y_train*prediction))

    weight = weightpara*weight.reshape((1036,1))
    weight = (weight/sum(weight)).reshape(1036)
    prefortest = np.zeros((1000,1))
    prefortrain = np.zeros((1036, 1))

    for i in range(T+1):
        prefortest = np.add(prefortest , alpha[i]*np.sign(np.dot(x_testc,w[i])))
    prefortest = np.sign(prefortest)
    for i in range(T+1):
        prefortrain = np.add(prefortrain , alpha[i]*np.sign(np.dot(x_trainc,w[i])))
    prefortrain = np.sign(prefortrain)
    trainerror[T] = np.sum(1-prefortrain*y_train)/2/1036
    testerror[T] = np.sum(1-prefortest * y_test) / 2 / 1000
    for i in range(T + 1):
        traerrup[i] = exp(-2*np.sum(np.power(0.5 - error[:i],2)))

    if T%100 == 0:
        print T


plt.plot(xrange(rounds),trainerror,label = 'training error')
plt.plot(xrange(rounds),testerror,label = 'test error')
plt.plot(xrange(rounds),traerrup,label = 'training error upperbound')
plt.legend(loc=1)
plt.show()

plt.hist(index,bins=1036)
plt.show()

plt.plot(xrange(rounds),error,label = 'errort')
plt.legend(loc=1)
plt.show()

plt.plot(xrange(rounds),alpha,label = 'alphat')
plt.legend(loc=1)
plt.show()