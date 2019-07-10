import matplotlib.pyplot as plt
from sklearn import datasets
import math
import numpy as np

'''
softmax分类器

'''

iris = datasets.load_iris()
data = iris.data[0:140]
target = iris.target[0:140]

m,n = data.shape
type = 3
x = np.array(data)
y = np.zeros((m,type),dtype=int)
for i in range(m):
	y[i][target[i]] = 1
testx = np.array(iris.data[141:150])
testy = iris.target[141:150]

w = np.random.randn(n,type)
b = np.zeros((1,type))

rate = 2e-1

for i in range(10000):
	score = np.dot(x,w) + b
	exp_score = np.exp(score)
	probs = exp_score / np.sum(exp_score,axis=1,keepdims=True)

	log = -np.log(probs[range(m),target])

	loss = np.sum(log) / m

	if i%10 == 0:
		print("iteration %4d loss: %f" % (i, loss))
	l = probs
	l[range(m),target]-=1#分类正确时，误差会减少，所以-1
	l /= m

	dw = np.dot(x.T,l)
	db = np.sum(l,axis=0,keepdims=True)

	w+= -rate * dw
	b+= -rate * db

a1 = np.dot(testx,w)+b

def softmax(v):
	exp = np.exp(v)
	return exp/np.sum(exp,axis=1,keepdims=True)

print(np.argmax(a1,axis=1))
print(softmax(a1))
print(testy)
