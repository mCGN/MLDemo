

import numpy as np

def sigmod(z):
	return 1 / (1 + np.exp(-z))

a = np.ones((2,3))
b = np.array([1,2,3]).reshape((1,3))

j = sigmod(a) * (1 - sigmod(a))
k = sigmod(a).T.dot(1 - sigmod(a))
#print(j)
#print(k)
#print([0.0] * a)

mc = np.zeros((2,3),dtype='uint8')

mp = np.ones((2,3))
print(mp)
mc[0][2] = 1
mc[1,1] = 1
v =[2,1]
reg = range(2)
mp[range(2),v]-=1
print(mp)