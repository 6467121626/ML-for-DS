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

def gaussian4(b,sigma):
    K = lambda x, y: exp(-np.sum((x - y) * (x - y)) / b)
    Kn = np.empty((350,350))
    for i in range(350):
        for j in range(350):
            Kn[i,j] = K(x_train[i,3],x_train[j,3])
    Kndinv = np.linalg.inv(Kn + np.diag(ones(350)) * sigma)
    Kny = np.dot(Kndinv, y_train)
    KDn = np.empty((350,350))
    for i in range(350):
        for j in range(350):
           KDn[i,j] = K(x_train[i,3],x_train[j,3])
    mean = np.dot(KDn,Kny)
    var = np.ones(350)*(sigma+1) - np.diag(np.dot(np.dot(KDn,Kndinv),np.transpose(KDn)))
    return mean,var

def gaussian(b,sigma):
    K = lambda x, y: exp(-np.sum((x - y) * (x - y)) / b)
    Kn = np.empty((350,350))
    for i in range(350):
        for j in range(350):
            Kn[i,j] = K(x_train[i],x_train[j])
    Kndinv = np.linalg.inv(Kn + np.diag(ones(350)) * sigma)
    Kny = np.dot(Kndinv, y_train)
    KDn = np.empty((42,350))
    for i in range(42):
        for j in range(350):
           KDn[i,j] = K(x_test[i],x_train[j])
    mean = np.dot(KDn,Kny)
    var = np.ones(42)*(sigma+1) - np.diag(np.dot(np.dot(KDn,Kndinv),np.transpose(KDn)))
    return mean,var

meant,vart = gaussian(5,0.1)
print(meant.flatten(),vart)

for b in range(5,17,2):
    for sigma in range(1,11):
        mean,var = gaussian(b,sigma/10.0)
        RMSE = sqrt(np.sum((mean - y_test)**2)/42)
        print(RMSE,b,sigma/10.0)

mean4,var4 = gaussian4(5,2)
index = np.argsort(x_train[:,3].flatten())
plt.plot(x_train[:,3],y_train,'g^',label='data')
plt.plot(x_train[:,3][index],mean4[index],'r',label='gaussian mean')
plt.legend(loc=1)
plt.show()

