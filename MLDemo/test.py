

import numpy as np

def sigmod(z):
	return 1 / (1 + np.exp(-z))

a = np.ones((2,3))
b = np.array([1,2,3]).reshape((1,3))

j = sigmod(a)*(1-sigmod(a))
k = sigmod(a).T.dot(1-sigmod(a))
print(j)
print(k)

print([0.0]*a)