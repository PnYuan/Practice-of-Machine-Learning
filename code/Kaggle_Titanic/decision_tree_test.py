"""
@author: PnYuan
@time: 2018-04-25

In this file, i attempt using decision tree for the task 'Titanic Disaster',
works are made as follows:
    1. data exploration with visualization.
    2. data pre-processing for tree model.
    3. using cross-validation to get suitable model hyperparameters.  
    4. training a decision tree.(Gini index).
    5. visualising the final model.
"""

##================== packages ==================##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

##================== file path ==================##
file_path_train = '../../data/raw_data/train.csv'
file_path_test = '../../data/raw_data/test.csv'

##================== load data with glimpse ==================##
df_train_org = pd.read_csv(file_path_train)
df_test_org = pd.read_csv(file_path_test)

# see data_pre_analysis.py

##================== feature construction ==================##
"""
feature:

"""

df_train = df_train_org.copy()
df_test = df_test_org.copy()
df_all = [df_train, df_test]

# encoding
for df in df_all:
    # Sex
    df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Embarked
    df['Embarked'] = df['Embarked'].fillna('unknown') 
    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2, 'unknown': -1} ).astype(int)
    
    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch']
    
    # Whether is alone
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 0, 'IsAlone'] = 1

    # Fare
    df['Fare'] = df['Fare'].fillna(-1) 
    df.loc[df['Fare'] <= 10, 'Fare'] = 0
    df.loc[(df['Fare'] > 10) & (df['Fare'] <= 50), 'Fare'] = 1
    df.loc[(df['Fare'] > 50) & (df['Fare'] <= 100), 'Fare'] = 2
    df.loc[df['Fare'] > 100, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)
    
    # Age
    df['Age'] = df['Age'].fillna(-1) 
    df.loc[(df['Age'] > 0)  & (df['Age'] <= 16), 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age'] = 4
    df['Age'] = df['Age'].astype('int')
    
# drop the redundant features
drop_elements = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']
df_train = df_train.drop(drop_elements, axis = 1)
df_test = df_test.drop(drop_elements, axis = 1)  
# df_train.columns
# df_test.columns

'''
# check the feature's correlation (Pearson correlation) 
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(df_train.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True)
plt.show()
'''

##================== model training ==================##

# using k-folds cross-validation to find the best tree depth
cv = KFold(n_splits = 10)
accuracies = list()
max_attributes = len(df_train.columns.tolist()) - 1
depth_range = range(1, max_attributes + 1)

accuracies = []
for depth in depth_range:
    accuracy = []
    tree_clf = tree.DecisionTreeClassifier(max_depth = depth)
    
    print("Current max depth: ", depth)
    
    for train_fold, valid_fold in cv.split(df_train):
        df_train_tmp = df_train.loc[train_fold]
        df_valid_tmp = df_train.loc[valid_fold]
        tree_clf_tmp = tree_clf.fit(X = df_train_tmp.drop(['PassengerId', 'Survived'], axis=1), y = df_train_tmp["Survived"])
        valid_acc = tree_clf_tmp.score(X = df_valid_tmp.drop(['PassengerId', 'Survived'], axis=1), y = df_valid_tmp["Survived"])
    
        accuracy.append(valid_acc)
        
    acc = sum(accuracy)/len(accuracy)
    accuracies.append(acc)    
    
fig = plt.figure()
plt.plot(depth_range, accuracies)
plt.title('accuracy with max_depth')
plt.xlabel('max_depth')
plt.ylabel('accuracy score')
plt.grid(True)
plt.ylim([0.7,0.85])
plt.show()


feature_list = df_train.drop(['PassengerId', 'Survived'], axis=1).columns.tolist()
X_train = df_train[feature_list]
y_train = df_train["Survived"]
X_test = df_test[feature_list]

# training
max_depth = 3
tree_clf = tree.DecisionTreeClassifier(max_depth = max_depth)
tree_clf.fit(X = X_train, y = y_train)

##================== predicting to get the submission .csv files ==================##
y_pred = tree_clf.predict(X_test)

df_submission = pd.DataFrame({
        "PassengerId":  df_test['PassengerId'],
        "Survived": y_pred
    })

df_submission.to_csv('submission.csv', index=False)

##================== visualising the final tree model (using graphviz)==================##
# Export our trained model as a .dot file
with open("tree.dot", 'w') as f:
     f = tree.export_graphviz(tree_clf,
                              out_file=f,
                              max_depth = max_depth,
                              impurity = True,
                              feature_names = list(feature_list),
                              class_names = ['Died', 'Survived'],
                              rounded = True,
                              filled= True )
        
#Convert .dot to .png
check_call(['dot','-Tpng','tree.dot','-o','tree.png'], shell=True)

print(' - PY131 - ')