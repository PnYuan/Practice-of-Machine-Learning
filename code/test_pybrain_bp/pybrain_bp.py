#-*- coding: utf-8 -*
    
'''''
@author: PY131
'''''

'''
preparation of data
'''
from sklearn import datasets  
iris_ds = datasets.load_iris()


X, y = iris_ds.data, iris_ds.target
label = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

from pybrain.datasets import ClassificationDataSet
# 4 input attributes, 1 output with 3 class labels
ds = ClassificationDataSet(4, 1, nb_classes=3, class_labels=label)  
for i in range(len(y)): 
    ds.appendLinked(X[i], y[i])
ds.calculateStatistics()

# split training, testing, validation data set (proportion 4:1)
tstdata_temp, trndata_temp = ds.splitWithProportion(0.25)  
tstdata = ClassificationDataSet(4, 1, nb_classes=3, class_labels=label)
for n in range(0, tstdata_temp.getLength()):
    tstdata.appendLinked( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

trndata = ClassificationDataSet(4, 1, nb_classes=3, class_labels=label)
for n in range(0, trndata_temp.getLength()):
    trndata.appendLinked( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

'''
implementation of BP network
'''
# build network 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer

n_h = 5
# 4 input nodes, 3 output node each represent one class
# here we set 5 hidden layer nodes.
# SoftmaxLayer(0/1) for multi-label output activation function
net = buildNetwork(4, n_h, 3, outclass = SoftmaxLayer)  

# standard BP algorithm
from pybrain.supervised.trainers import BackpropTrainer

# standard(incremental) BP algorithm: 
# trainer = BackpropTrainer(net, trndata)
# trainer.trainEpochs(1)

# accumulative BP algorithm: 
trainer = BackpropTrainer(net, trndata, batchlearning=False)
err_train, err_valid = trainer.trainUntilConvergence(maxEpochs=50)

# # default validationProportion=0.25
# import datetime
# starttime = datetime.datetime.now()
# 
# # err_train, err_valid = trainer.trainUntilConvergence(maxEpochs=0)
# 
# endtime = datetime.datetime.now()
# runtime = (endtime - starttime).seconds + (endtime - starttime).microseconds/1000000 
# print( "runtime: %.3f ms " % runtime)

'''
test of model
'''
# convergence curve 
import matplotlib.pyplot as plt
plt.plot(err_train,'b',err_valid,'r')
plt.title('BP network classification')  
plt.ylabel('accuracy')  
plt.xlabel('epochs')  
plt.show()

# model testing
from pybrain.utilities import percentError
tstresult = percentError( trainer.testOnClassData(), tstdata['target'] )
print("epoch: %4d" % trainer.totalepochs, " test error: %5.2f%%" % tstresult)

'''
visualization
'''
import numpy as np
import matplotlib.pyplot as plt

f1 = plt.figure(2) 
h = 0.5
x0_min = np.zeros(4)
x0_max = np.zeros(4)
x0 = np.zeros(4)
for i in range(4):
    x0_min[i], x0_max[i] = X[:, i].min()-1, X[:, i].max()+1
    
x0, x1, x2, x3 = np.meshgrid(np.arange(x0_min[0], x0_max[0], h),
                 np.arange(x0_min[1], x0_max[1], h),
                 np.arange(x0_min[2], x0_max[2], h),
                 np.arange(x0_min[3], x0_max[3], h))

griddata = ClassificationDataSet(4, 1, nb_classes=3, class_labels=label)
for i in range(X.size):
    griddata.addSample([x0.ravel()[i],
                        x1.ravel()[i],
                        x2.ravel()[i],
                        x3.ravel()[i]], [0])

griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy

out = net.activateOnDataset(griddata)
out = out.argmax(axis=1)  # the highest output activation gives the class
out = out.reshape(X.shape)

colors = ['b', 'g', 'r']

for c in [0,1,2]:
    here, _ = np.where(tstdata['class']==c)
    # here we plot only first two attributes in a 2D map
    plt.title('iris classification using BP network')  
    plt.xlabel('sepal length')  
    plt.ylabel('sepal width')  
    plt.scatter(tstdata['input'][here,0], tstdata['input'][here,1], color=colors[c], marker='o', s=50, label=label[c])
    
# plt.contourf(x1[:,:,0,0], x2[:,:,0,0], out[0:2])   # plot the decision area in 2D map of first two attributes
plt.legend(loc='upper right')
plt.show()

 