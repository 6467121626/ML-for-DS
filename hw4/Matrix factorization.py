import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
rating_train = np.array(pd.read_csv('ratings.csv',header = -1))
rating_test = np.array(pd.read_csv('ratings_test.csv',header = -1))
f  = open('movies.txt', 'r')
movie = f.readlines()
M = np.zeros((943,1682))
for items in rating_train:
    M[np.int32(items[0]) - 1, np.int32(items[1]) - 1] = items[2]
def obj(U, V ,M):
    sumu = np.sum(U ** 2) / 2
    sumv = np.sum(V ** 2) / 2
    sumall = 0
    for items in rating_train:
        sumall = sumall + (M[np.int32(items[0]) - 1, np.int32(items[1]) - 1] - np.dot(U[np.int32(items[0]) - 1],
                  V[:,np.int32(items[1]) - 1])) ** 2 * 2
    sumall = sumall + sumu + sumv
    return -sumall
def calrmse(U,V):
    sumvalue = 0
    for items in rating_test:
        sumvalue = sumvalue +(items[2]-np.dot(U[np.int32(items[0]) - 1],
                  V[:,np.int32(items[1]) - 1]))**2
    return np.sqrt(sumvalue/5000)
def findmovie(v):
    v_starwars = np.sum(np.subtract(v, v[:,49].reshape((10, 1)))**2,axis=0)
    v_lady = np.sum(np.subtract(v, v[:, 484].reshape((10, 1)))**2,axis=0)
    v_goodfellas = np.sum(np.subtract(v, v[:, 181].reshape((10, 1)))**2,axis=0)
    movie_starwars = []
    movie_lady = []
    movie_goodfellas = []
    index_starwars = np.argsort(v_starwars)
    index_lady = np.argsort(v_lady)
    index_goodfellas = np.argsort(v_goodfellas)
    for items in index_starwars[1:11]:
        movie_starwars.append(movie[items])
    for items in index_lady[1:11]:
        movie_lady.append(movie[items])
    for items in index_goodfellas[1:11]:
        movie_goodfellas.append(movie[items])
    return movie_starwars,v_starwars[index_starwars][1:11],movie_lady,v_lady[index_lady][1:11],movie_goodfellas,v_goodfellas[index_goodfellas][1:11]
def MF():
    num_itereation = 100
    v = np.empty((10,1682))
    u = np.empty((943,10))
    for i in xrange(1682):
        v[:,i] = np.random.multivariate_normal(np.zeros(10), np.diag(np.ones(10)), 1)
    for i in xrange(943):
        u[i,:] = np.random.multivariate_normal(np.zeros(10), np.diag(np.ones(10)), 1)


    for k in xrange(num_itereation):
        for i in xrange(943):
            index1 = M[i,:] != 0
            u[i,:] =  np.dot(np.linalg.inv(np.diag(np.ones(10)*0.25)+np.dot(v[:,index1],v[:,index1].T)),
                         np.dot(M[i,:][index1],v[:,index1].T))
        for j in xrange(1682):
            index2 = M[:, j] != 0
            v[:, j] = np.dot(np.linalg.inv(np.diag(np.ones(10) * 0.25) + np.dot(u[index2].T, u[index2])),
                     np.dot(M[:, j][index2], u[index2]))
        if k%10 == 0:
            print k
    return [obj(u,v,M),calrmse(u,v)],v

v_new = np.empty((10, 1682))
v_max = np.empty((10, 1682))
ro = np.empty((10,2))
for i in xrange(10):
    print "iteration:%d"%i
    ro[i],v_new = MF()
    if ro[i,0] >= np.max(ro[:i+1,0]):
        v_max = v_new


indexs = np.argsort(-ro[:,0])
print ro[indexs]
print findmovie(v_max)

