#-*- coding: utf-8 -*
    
'''''
@author: PY131, created on 17.4.24

here we use iris data set to conduct an experiment
'''''

'''
get data
'''
from sklearn.datasets import make_circles
# import numpy as np
import matplotlib.pyplot as plt 

X, y = make_circles(100, noise=0.05)  # 2 input 1 output
f1 = plt.figure(1) 
plt.scatter(X[:,0], X[:,1], s=40, c=y)
plt.title("circles data")
# plt.show()


'''
BP implementation
'''
from BP_network import *
import matplotlib.pyplot as plt 

nn = BP_network()  # build a BP network class
nn.CreateNN(2, 6, 1, 'Sigmoid')  # build the network

e = []
for i in range(2000): 
    err, err_k = nn.TrainStandard(X, y.reshape(len(y),1), lr=0.5)
    e.append(err)
f2 = plt.figure(2) 
plt.xlabel("epochs")
plt.ylabel("accumulated error")
plt.title("circles convergence curve")
plt.plot(e)
# plt.show()

'''
draw decision boundary
'''
import numpy as np
import matplotlib.pyplot as plt 

h = 0.01
x0_min, x0_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
x1_min, x1_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, h),
                     np.arange(x1_min, x1_max, h))

f3 = plt.figure(3)
z = nn.PredLabel(np.c_[x0.ravel(), x1.ravel()])
z = z.reshape(x0.shape)

plt.contourf(x0, x1, z, cmap = plt.cm.Paired)
plt.scatter(X[:,0], X[:,1], s=40, c=y)
plt.title("circles classification")
plt.show()








