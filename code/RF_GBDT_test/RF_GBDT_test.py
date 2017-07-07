# coding = <utf-8>

'''
@author: PY131
'''
# for visualization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# for data
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# for classifier model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

########## 1. getting data set ##########
ds_iris = load_iris()
ds_breast_cancer = load_breast_cancer()

datasets = [[ds_iris.data, ds_iris.target],
            [ds_breast_cancer.data, ds_breast_cancer.target]]
datasets_names = ['iris',
                  'breast_cancer']

fig_num = 0
# draw sactter
f1 = plt.figure(fig_num)
for i, ds in enumerate(datasets):
    X, y = ds 
    cm_bright = ListedColormap(['r', 'b', 'g'])
    ax = plt.subplot(1, 2, i+1)
    ax.set_title(datasets_names[i])   
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright, edgecolors='k')  
plt.show()
    
########## 2. training and testing ##########

##### 2.1 Random Forest #####

### 2.1.1 Parameter testing: the different max_depth for each based tree
fig_num +=1
f2 = plt.figure(fig_num)
for i, ds in enumerate(datasets):
    #split the data set
    X, y = ds 
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    
    x = []
    scores = []
    for md in range(1, 50, 1):
        x.append(md)
        
        # test
        clf = RandomForestClassifier(n_estimators=10,
                                     max_depth=md,
                                     max_features='log2', 
                                     bootstrap=True)
        # training and testing
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
        
    # plot the result as 
    ax = plt.subplot(1, len(datasets), i+1)
    ax.set_title('RF for %s' % datasets_names[i])
    ax.set_xlabel('max_depth')
    ax.set_ylabel('accuracy_scores')
    ax.set_ylim([0, 1])
    ax.plot(x, scores)
plt.show()
  
### 2.1.2 Parameter testing: the different n_estimators for each based tree
fig_num += 1
f = plt.figure(fig_num)
for i, ds in enumerate(datasets):
    #split the data set
    X, y = ds 
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    
    x = []
    scores = []
    for nt in range(1, 200, 1):
        x.append(nt)
        
        # test
        clf = RandomForestClassifier(n_estimators=nt,
                                     max_depth=2,
                                     max_features='log2', 
                                     bootstrap=True)
        # training and testing
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
        
    # plot the result as 
    ax = plt.subplot(1, len(datasets), i+1)
    ax.set_title('RF for %s' % datasets_names[i])
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('accuracy_scores')
    ax.set_ylim([0, 1])
    ax.plot(x, scores)
plt.show()


##### 2.2 GBDT #####

### 2.2.1 Parameter testing: the different max_depth for each base tree
fig_num += 1
f = plt.figure(fig_num)
for i, ds in enumerate(datasets):
    #split the data set
    X, y = ds 
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    
    x = []
    scores = []
    for md in range(1, 20, 1):
        x.append(md)
        clf = GradientBoostingClassifier(n_estimators=50, 
                                         max_depth=md,
                                         learning_rate=0.01,
                                         subsample=0.5,
                                         min_samples_leaf=1)
        # training and testing
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    
    # plot the result as 
    ax = plt.subplot(1, len(datasets), i+1)
    ax.set_title('GBDT for %s' % datasets_names[i])
    ax.set_xlabel('max_depth')
    ax.set_ylabel('accuracy_scores')
    ax.set_ylim([0, 1])
    ax.plot(x, scores)
plt.show()

### 2.2.2 Parameter testing: the different n_estimators for iterative epochs
fig_num += 1
f = plt.figure(fig_num)
for i, ds in enumerate(datasets):
    #split the data set
    X, y = ds 
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    
    x = []
    scores = []
    for nt in range(1, 100, 1):
        x.append(nt)
        clf = GradientBoostingClassifier(n_estimators=nt, 
                                         max_depth=1,
                                         learning_rate=0.01,
                                         subsample=0.5,
                                         min_samples_leaf=1)
        # training and testing
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))
    
    # plot the result as 
    ax = plt.subplot(1, len(datasets), i+1)
    ax.set_title('GBDT for %s' % datasets_names[i])
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('accuracy_scores')
    ax.set_ylim([0, 1])
    ax.plot(x, scores)
plt.show()



print(' - PY131 -')