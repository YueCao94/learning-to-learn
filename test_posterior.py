import numpy as np
import scipy
import posterior

def rosen(x):
	return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


x =np.zeros((2))

for i in range(8):
	for j in range(8):
		x[0]=i
		x[1]=j


sample={
	'x':[],
	'y':[]
}


t = np.random.uniform(-10, 10, (20,2))

sample['x'] = t

for i in range(20):
	sample['y'].append(rosen(t[i]))


pos = posterior.posterior()
pos.fit(sample)

for i in range(8):
	for j in range(8):
		x[0]=i
		x[1]=j
		print (i,j, rosen(x), pos.kriging(x))





