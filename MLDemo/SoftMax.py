import numpy as np
import matplotlib.pyplot as plot
import pandas

text = np.array(pandas.read_csv('data/iris.data'))
#print(text)
data = []#m*n
#数据处理
for x in range(len(text)):
	label = 0
	l = (text[x][4])
	if(l == 'Iris-setosa'):
		label = 0
	elif (l == 'Iris-versicolor'):
		label = 1
	elif (l == 'Iris-virginica'):
		label = 2
	data.append([text[x][0],text[x][1],text[x][2],text[x][3],label])

data = np.array(data)

k = 3
m = len(data)
n = 4
alpha = 1e-4

theta = np.ones((k,n))#
trainset = data[:,:-1]
print(data[:,-1])
labelset = data[:,-1].reshape(m,1)
def softmax(x):
	return np.exp(x) / np.sum(np.exp(x))

def j(data):
    pass

i = 0
print(labelset)
print(softmax(trainset.dot(theta.T))*labelset)

#print(np.array([[1,3,4,5],[3,4,5,6]])*np.array([[0],[1]]))
'''

while i < 10000:
	sum = 0
	for j in range(0,k):
		softmax(data[:,-1]*theta)
	i = i + 1

'''

def getlabel(x):
    for v in x.ite:
        pass