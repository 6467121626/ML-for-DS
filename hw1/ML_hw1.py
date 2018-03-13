import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

x_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
x_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

x = np.mat(np.array(x_train))
y = np.mat(np.array(y_train))
I = np.mat(np.eye(7))

U, s, V = np.linalg.svd(x, full_matrices=True)
W_RR = np.ones((5000,7))
df = np.ones((5000))

for k in range(5000):
    W_RR[k] = np.dot(np.dot((k*I + np.dot(np.transpose(x),x)).I,np.transpose(x)),y).reshape(7)
    s_lambda = s*s / (s*s + k)

    df[k] = s_lambda.sum()

for k in range(7):
    plt.plot(df,W_RR[:,k],label="feature:%d"%(k+1))
legend(loc='upper left')
plt.show()

x_test = np.array(x_test)
y_test = np.array(y_test)
RMSE2 = np.ones(51)

for k in range(51):
    y_prediction = np.sum(x_test*W_RR[k],axis=1)
    error = y_prediction - y_test.reshape(42)
    error_square = error*error/42
    RMSE2[k] = error_square.sum()

print(y_prediction)

plt.plot(range(51),RMSE2)
plt.xlabel('lambda')
plt.ylabel('RMSE')
plt.show()

x_train2 = np.ones((350,13))
x_train3 = np.ones((350,19))
print(x_train2[:,7:13].shape)
x_train_2 = np.array(x_train)*np.array(x_train)
x_train_3 = np.array(x_train)*np.array(x_train)*np.array(x_train)
x_train2[:,:7] = np.array(x_train)
x_train2[:,7:13] = x_train_2[:,:6]

x_train3[:,:7] = np.array(x_train)
x_train3[:,7:13] = x_train_2[:,:6]
x_train3[:,13:] = x_train_3[:,:6]

W_RR2 = np.ones((501,13))
W_RR3 = np.ones((501,19))
I2 = np.mat(np.eye(13))
I3 = np.mat(np.eye(19))


for k in range(501):
    W_RR2[k] = np.dot(np.dot((k*I2 + np.dot(np.transpose(x_train2),x_train2)).I,
                             np.transpose(x_train2)),y).reshape(13)



for k in range(501):
    W_RR3[k] = np.dot(np.dot((k*I3 + np.dot(np.transpose(x_train3),x_train3)).I,
                             np.transpose(x_train3)),y).reshape(19)


x_test2 = np.ones((42,13))
x_test3 = np.ones((42,19))
x_test_2 = np.array(x_test)*np.array(x_test)
x_test_3 = np.array(x_test)*np.array(x_test)*np.array(x_test)
x_test2[:,:7] = x_test
x_test2[:,7:] = x_test_2[:,:6]
x_test3[:,:7] = x_test
x_test3[:,7:13] = x_test_2[:,:6]
x_test3[:,13:] = x_test_3[:,:6]

RMSE21 = np.ones(501)
RMSE22 = np.ones(501)
RMSE23 = np.ones(501)
for k in range(501):
    y_prediction = np.sum(x_test*W_RR[k],axis=1)
    error = y_prediction - y_test.reshape(42)
    error_square = error*error/42
    RMSE21[k] = error_square.sum()

for k in range(501):
    y_prediction = np.sum(x_test2*W_RR2[k],axis=1)
    error = y_prediction - y_test.reshape(42)
    error_square = error*error/42
    RMSE22[k] = error_square.sum()

for k in range(501):
    y_prediction = np.sum(x_test3*W_RR3[k],axis=1)
    error = y_prediction - y_test.reshape(42)
    error_square = error*error/42
    RMSE23[k] = error_square.sum()


plt.plot(range(501),RMSE21,label="first-order")
plt.plot(range(501),RMSE22,label="second-order")
plt.plot(range(501),RMSE23,label="third-order")
plt.xlabel('lambda')
plt.ylabel('RMSE')
legend(loc='upper left')
plt.show()

