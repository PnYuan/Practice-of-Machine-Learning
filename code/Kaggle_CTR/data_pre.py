"""
data preparation for model-based task:
    
    1. extract the data with selected features;
    2. set the rare categorical values to 'other';
    3. fit a label encoder and a one-hot encoder for new data set

"""

##==================== Package ====================##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from dummyPy import OneHotEncoder
import random

import pickle  # to store temporary variable

##==================== File-Path (fp) ====================##
## raw data (for read)
fp_train = "../data/train.csv"
fp_test  = "../data/test.csv"

## subsample training set
fp_sub_train_f = "../data/sub_train_f.csv"

fp_col_counts = "../data/col_counts"

## data after selecting features (LR_fun needed)
## and setting rare categories' value to 'other' (feature filtering)
fp_train_f = "../data/train_f.csv"
fp_test_f  = "../data/test_f.csv"

## storing encoder for labeling / one-hot encoding task
fp_lb_enc = "../data/lb_enc"
fp_oh_enc = "../data/oh_enc"

##==================== pre-Processing ====================##
## some simple original features is selected for dataset
'''features are used
    C1:           int,     1001, 1002, ...
    banner_pos:   int,     0,1,2,3,...
    site_domain:  object,  large set of object variables 
    site_id:      object,  large set of object variables 
    site_category:object,  large set of object variables 
    app_id:       object,  large set of object variables 
    app_category: object,  small set of object variables
    device_type:  int,     0,1,2,3,4
    device_conn_type:int,  0,1,2,3
    C14:          int,     small set of int variables
    C15:          int,     ...
    C16:          int,     ...
'''
## feature names
cols = ['C1', 
        'banner_pos', 
        'site_domain', 
        'site_id',
        'site_category',
        'app_id',
        'app_category', 
        'device_type', 
        'device_conn_type',
        'C14', 
        'C15',
        'C16']

cols_train = ['id', 'click']
cols_test  = ['id']
cols_train.extend(cols)
cols_test.extend(cols)

## data reading
df_train_ini = pd.read_csv(fp_train, nrows = 10)
df_train_org = pd.read_csv(fp_train, chunksize = 1000000, iterator = True)
df_test_org  = pd.read_csv(fp_test,  chunksize = 1000000, iterator = True)

#----- counting features' categories numbers -----#
## 1.init_dict
cols_counts = {}  # the categories count for each feature
for col in cols:
    cols_counts[col] = df_train_ini[col].value_counts()

## 2.counting through train-set
for chunk in df_train_org:
    for col in cols:
        cols_counts[col] = cols_counts[col].append(chunk[col].value_counts())

## 3.counting through test-set
for chunk in df_test_org:
    for col in cols:
        cols_counts[col] = cols_counts[col].append(chunk[col].value_counts())
        
## 4.merge the deduplicates index in counting vectors
for col in cols:
    cols_counts[col] = cols_counts[col].groupby(cols_counts[col].index).sum()
    # sort the counts
    cols_counts[col] = cols_counts[col].sort_values(ascending=False)   

## 5.store the value_counting
pickle.dump(cols_counts, open(fp_col_counts, 'wb'))

## 6.show the distribution of value_counts
fig = plt.figure(1)
for i, col in enumerate(cols):
    ax = fig.add_subplot(4, 3, i+1)
    ax.fill_between(np.arange(len(cols_counts[col])), cols_counts[col].get_values())
    # ax.set_title(col)
plt.show()

#----- set rare to 'other' -----#
# cols_counts = pickle.load(open(fp_col_counts, 'rb'))

## save at most k indices of the categorical variables
## and set the rest to 'other'
k = 99
col_index = {}
for col in cols:
    col_index[col] = cols_counts[col][0: k].index

df_train_org = pd.read_csv(fp_train, dtype = {'id': str}, chunksize = 1000000, iterator = True)
df_test_org  = pd.read_csv(fp_test,  dtype = {'id': str}, chunksize = 1000000, iterator = True)

## train set
hd_flag = True  # add column names at 1-st row
for chunk in df_train_org:
    df = chunk.copy()
    for col in cols:
        df[col] = df[col].astype('object')
        # assign all the rare variables as 'other'
        df.loc[~df[col].isin(col_index[col]), col] = 'other'
    with open(fp_train_f, 'a') as f:
        df.to_csv(f, columns = cols_train, header = hd_flag, index = False)
    hd_flag = False

## test set
hd_flag = True  # add column names at 1-st row
for chunk in df_test_org:
    df = chunk.copy()
    for col in cols:
        df[col] = df[col].astype('object')
        # assign all the rare variables as 'other'
        df.loc[~df[col].isin(col_index[col]), col] = 'other'
    with open(fp_test_f, 'a') as f:
        df.to_csv(f, columns = cols_test, header = hd_flag, index = False)      
    hd_flag = False    

#----- generate encoder for label encoding -----#
#----- generate encoder for one-hot encoding -----#
'''
notes: here we do not apply label/one-hot transform
       as we do it later in the iteration of model training on chunks
'''
## 1.label encoding
lb_enc = {}
for col in cols:
    col_index[col] = np.append(col_index[col], 'other')

for col in cols:
    lb_enc[col] = LabelEncoder()
    lb_enc[col].fit(col_index[col])
    
## store the label encoder
pickle.dump(lb_enc, open(fp_lb_enc, 'wb'))

## 2.one-hot encoding
oh_enc = OneHotEncoder(cols)

df_train_f = pd.read_csv(fp_train_f, index_col=None, chunksize=500000, iterator=True)
df_test_f  = pd.read_csv(fp_test_f, index_col=None, chunksize=500000, iterator=True)

for chunk in df_train_f:
    oh_enc.fit(chunk)
for chunk in df_test_f:
    oh_enc.fit(chunk)
    
## store the one-hot encoder
pickle.dump(oh_enc, open(fp_oh_enc, 'wb'))

#----- construct of original train set (sub-sampling randomly) -----#
n = sum(1 for line in open(fp_train_f)) - 1  # total size of train data (about 46M)
s = 2000000 # desired train set size (2M)

## the 0-indexed header will not be included in the skip list
skip = sorted(random.sample(range(1, n+1), n-s)) 
df_train = pd.read_csv(fp_train_f, skiprows = skip)
df_train.columns = cols_train

## store the sub-sampling train set as .csv
df_train.to_csv(fp_sub_train_f, index=False) 

print(' - PY131 - ')