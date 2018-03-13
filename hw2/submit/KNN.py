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

def dis(arr1,arr2):
    arr = abs(arr1 - arr2)
    result = sum(arr)
    return result
distances = zeros(x_train.shape[0])
prediction = zeros((x_test.shape[0],20))
for i in xrange(x_test.shape[0]):
    for j in xrange(distances.shape[0]):
        distances[j] = dis(x_test[i,:],x_train[j,:])
    index = np.argsort(distances)
    for k in xrange(20):
        prediction[i][k] = round(sum(y_train[index[:k+1]])/(k+1))
accuracy = sum((prediction * y_test + (1 - prediction)*(1 - y_test)),axis=0)/x_test.shape[0]
plt.plot(xrange(1,21),accuracy)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.show()
print accuracy