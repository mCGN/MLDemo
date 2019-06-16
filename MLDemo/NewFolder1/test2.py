import matplotlib.pyplot as plt
from sklearn import datasets
import math
import numpy as np

#输入是4维的向量（4），隐藏层3，输出是（3）
iris = datasets.load_iris()
x = iris.data
n = len(x)

y = np.zeros((n,3))
for i in range(len(iris.target)):
    y[i][iris.target[i]] = 1

W1 = np.ones((4,3))
W2 = np.ones((3,3))

def sigmod(z):
	return 1 / (1 + np.exp(-z))

def softmax(z):
	return np.exp(z) / np.sum(np.exp(z))

def propagation():
	a1 = sigmod(x.dot(W1))
	return softmax(a1.dot(W2))

def backPropagation():

	pass

#a = np.array([1,0,0,0,1,0]).reshape((2,3))

#print(softmax(a))




