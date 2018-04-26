"""
this is a file for data loading and pre-analysis

@author: PnYuan
@time: 2018-04-24
"""

##================== packages ==================##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##================== file path ==================##
file_path_train = '../../data/raw_data/train.csv'
file_path_test = '../../data/raw_data/test.csv'

##================== main process ==================##

#---------- load training data ----------#
df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)
# df_train.info()
# df_train.describe()
# df_train.head()

#---------- analysis the mapping feature to result  ----------#
sns.set(style = 'white')
fig = plt.figure()

# Pclass
plt.subplot2grid((3,3),(0,0))
ax = sns.countplot(x = 'Pclass', hue = 'Survived', data = df_train)
# plt.title('Pclass')
# plt.xlabel('Pclass') 
# plt.ylabel('count')
plt.grid(True)
# plt.show()

# Sex
plt.subplot2grid((3,3),(0,1))
ax = sns.countplot(x = 'Sex', hue = 'Survived', data = df_train)
# plt.title('Sex')
# plt.xlabel('Sex') 
# plt.ylabel('count')
plt.grid(True)
# plt.show()

# Embarked
plt.subplot2grid((3,3),(0,2))
ax = sns.countplot(x = 'Embarked', hue = 'Survived', data = df_train)
# plt.title('Embarked')
# plt.xlabel('Embarked') 
# plt.ylabel('count')
plt.grid(True)
# plt.show()

# SibSp
plt.subplot2grid((3,3),(1,0))
ax = sns.countplot(x = 'SibSp', hue = 'Survived', data = df_train)
# plt.title('SibSp')
# plt.xlabel('SibSp') 
# plt.ylabel('count')
plt.grid(True)
# plt.show()

# Parch
plt.subplot2grid((3,3),(1,1))
ax = sns.countplot(x = 'Parch', hue = 'Survived', data = df_train)
# plt.title('Parch')
# plt.xlabel('Parch') 
# plt.ylabel('count')
plt.grid(True)
# plt.show()

# Age

type = ['not-known', 'child', 'adult', 'elder']

df_train['Age_class'] = (df_train['Age'] <= 15).astype('int')
df_train['Age_class'] += ((df_train['Age'] > 15).astype('int') & (df_train['Age'] <= 55).astype('int')) * 2
df_train['Age_class'] += (df_train['Age'] > 55).astype('int') * 3
df_train['Age_class'] += (df_train['Age'].isnull()).astype('int') * 0

df_train['Age_class_n'] = df_train['Age_class']
df_train['Age_class_n'].astype('str')
for i in range(len(type)):
    df_train.loc[df_train['Age_class'] == i, 'Age_class_n'] = type[i]

plt.subplot2grid((3,3),(1,2))
ax = sns.countplot(x = 'Age_class_n', hue = 'Survived', data = df_train)
plt.grid(True)
# plt.show()

# Fare
fare_series = df_train['Fare']
df_train['Fare_class_n'] = pd.cut(fare_series, 10, labels=range(10, 500, 50))

plt.subplot2grid((3,3),(2,0), colspan=3)
ax = sns.countplot(x = 'Fare_class_n', hue = 'Survived', data = df_train)
plt.grid(True)
plt.show()

print(' - PY131 - ')