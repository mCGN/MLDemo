import pandas
import numpy as np
import matplotlib.pyplot as plt

data = pandas.read_csv('data/house.csv')
one = np.ones((len(data),1),'int32')
global dataset
dataset = np.array( np.hstack((one,data)))
#print(dataset)

theta= [1,1]
def h(theta,y):
    return np.dot(theta,y)

X = dataset[:,0:-1]
Y = dataset[:,-1]

#批量梯度下降
def method1():
    i = 0;
    maxnum = 100000
    alpha = 0.00001#学习速率
    error = [0,0]
    epsilon = 1e-5
    theta = [1,1]
    while i<maxnum:
        temp = np.dot((h(theta,dataset[:,0:-1].T)-dataset[:,-1]),dataset[:,0:-1])
        theta = theta - alpha*temp
        if np.linalg.norm(theta-error) <epsilon:
            return theta
        else:
            error = theta
        i=i+1
    return theta

#利用公式计算θ
def opt(X,y):
    A = np.linalg.inv(X.T.dot(X))
    B = X.T.dot(y)
    C = A.dot(B)
    return C

#theta = method1()
theta =opt(X,Y)
print(X)
#展示数据
plt.plot(dataset[:,1],dataset[:,2],'bo')
plt.plot(dataset[:,1],X.dot(theta),'r')
plt.show()