import numpy
import sys
import matplotlib.pyplot as plot
import pandas

#逻辑回归,二分类
dt = numpy.array(pandas.read_csv('data/logistics.csv',',')).reshape((-1,3))

m = dt.shape[0]

data = numpy.hstack((numpy.ones((m,1)),dt[:,0:-1]))#训练集
label = dt[:,-1].reshape(m,1)#标签

def sigmoid(z):
    return 1 / (1 + numpy.exp(-z))

def GradientDescent(data,label):
    m,n = data.shape
    theta = numpy.ones((n,1))
    alpha = 0.00001
    
    for x in range(100000):
        h = sigmoid(data.dot(theta))
        theta = theta + alpha * data.T.dot((label - h))
    return theta
    pass

theta = GradientDescent(data,label)
print(theta)
#print(theta.shape)
def main():
    x1 = []
    y1 = []
    x2 = []
    y2 = []

    for i in range(dt.shape[0]):
        if dt[i][2] == 0:
            x1.append(dt[i][0])
            y1.append(dt[i][1])
        else:
            x2.append(dt[i][0])
            y2.append(dt[i][1])

    plot.plot(x1,y1,c='red',marker='s',ls='None')
    plot.plot(x2,y2,c='black',marker='s',ls='None')
    x = numpy.arange(-1,1,0.1)
    y = (-theta[0,0]-theta[1,0]*x)/theta[2,0]
    plot.plot(x,y)
    plot.show()

    return 1
    pass

main()
