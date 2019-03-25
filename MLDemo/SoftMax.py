import numpy as np
import matplotlib.pyplot as plot
import pandas

text = np.array(pandas.read_csv('data/iris.data'))
#print(text)

data = []
#数据处理
for x in range(len(text)):
	label = 0
	if(text[x][4]=='Iris-setosa'):
		lebel = 0
	elif (text[x][4] == 'Iris-versicolor'):
		lebel = 1
	elif (text[x][4] == 'Iris-virginica'):
		lebel = 2
	data.append([text[x][0],text[x][1],text[x][2],text[x][3],label])


k = 3
m = len(data)
n = 4
alpha = 1e-4;

theta = numpy.ones((k,n+1))

def softmax(x):
	return np.exp(x)/np.sum(np.exp(x))

def j(data):
    pass

#j(θ)= Σ