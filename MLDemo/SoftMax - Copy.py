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

trainset = data[:,:-1]#m*n
k = 3
m,n = trainset.shape
alpha = 1e-4

labelset = data[:,-1].reshape(m,1)
def softmax(x):
	return np.exp(x) / np.sum(np.exp(x))

softmax()
def train():
	theta = np.ones((k,n))
	i = 0
	while i < 100:
	
		i+=1