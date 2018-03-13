import matplotlib.pyplot as plt
import numpy as np

def findc(x,c,num):
    a = []
    for index, item in enumerate(c):
        if item == num:
            a.append(x[index])
    return np.array(a)

def updatec(uk,x):
    cluster = np.empty(500)
    for index,item in enumerate(x):
        dist = np.sum((uk - item)**2,axis=1)
        cluster[index] = np.argmin(dist)
    return cluster

def updateuk(c,x,numc):
    u = np.empty((numc, 2))
    for i in range(numc):
        ci = findc(x, c, i)
        u[i, :] = np.average(ci, axis=0)
    return u


def plotshowdatas(x,c,numc):
    for i in range(numc):
        cudata = findc(x,c,i)
        plt.plot(cudata[:,1], cudata[:,0], 'x')
    plt.axis('equal')
    plt.title("k = %d"%numc)
    plt.show()


cov1 = [[1, 0], [0, 1]]
mean1 = [0, 0]
cov2 = [[1, 0], [0, 1]]
mean2 = [3, 0]
cov3 = [[1, 0], [0, 1]]
mean3 = [0, 3]
weight = [0.2, 0.5, 0.3]
a = np.random.choice(3,500,p=weight)
a , b = np.histogram(a,bins=3)
x1 , y1 = np.random.multivariate_normal(mean1, cov1, a[0]).T
x2 , y2 = np.random.multivariate_normal(mean2, cov2, a[1]).T
x3 , y3 = np.random.multivariate_normal(mean3, cov3, a[2]).T
data = np.empty((500,2))
data[:,0] = np.append(np.append(x1,x2),x3)
data[:,1] = np.append(np.append(y1,y2),y3)
# plt.plot(x1, y1, 'x')
# plt.plot(x2, y2, 'x')
# plt.plot(x3, y3, 'x')
#
# plt.axis('equal')
# plt.show()
# plt.plot(data[:,0], data[:,1], 'x')
# plt.show()
obj = np.empty((4,20))
for i in range(2,6):
    num_of_clusters = i
    c = np.empty(500)
    uk = (np.random.rand(num_of_clusters,2))*3
    # print uk-[1,1],uk,np.sum(uk,axis=0)
    # print np.array(findc(np.ones((6,2)),[1,23,4,2213,23,2],23))
    for j in range(20):
        c = updatec(uk,data)
        uk = updateuk(c,data,num_of_clusters)
        obj[i-2,j] = np.sum((data - uk[np.int8(c)])**2)
    if i == 3 or i ==5:
        plotshowdatas(data, c, num_of_clusters)
for i in range(4):
    plt.plot(range(20), obj[i], label = "K = %d" % (i+2) )
plt.legend(loc=1)
plt.xlabel('iterations')
plt.title("K-means object function")
plt.show()


