
import pandas
import numpy as np
import matplotlib.pyplot as plt
import sys

# 拟合曲线

def opt(X,y):
    A = np.linalg.inv(X.T.dot(X))
    B = X.T.dot(y)
    C = A.dot(B)
    return C

#ax^2+bx+c

x = np.arange(0,1,0.05)

y = x**3 + x**2+x+np.random.rand(len(x))
m=[]
for i in range(4):
    m.append(x**i)
X = np.array(m).T
Y = y.reshape(20,1)
theta = opt(X,Y)
print(theta)
plt.plot(x,y,color='m',linestyle='',marker='o')
plt.plot(x,X.dot(theta),color='g',linestyle='-',marker='')
plt.show()

