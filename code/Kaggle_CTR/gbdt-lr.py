"""
a demo of GBDT-LR-based Avazu-CTR prediction.
    
    1.apply label encoding for GBDT input;
    2.apply one-hot encoding for GBDT output (LR input);
    3.building GBDT-LR, training and tuning; 
    4.predicting to generate submission.
    
"""

##==================== Package ====================##
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier  # using SGDClassifier for training incrementally
from sklearn.preprocessing import LabelEncoder
from dummyPy import OneHotEncoder  # for one-hot encoding on a large scale of chunks
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
import pickle
import gc

##==================== File-Path (fp) ====================##
## data after selecting features (LR_fun needed)
## and setting rare categories' value to 'other' (feature filtering)
fp_train_f = "../data/train_f.csv"
fp_test_f  = "../data/test_f.csv"
## subsample training set
fp_sub_train_f = "../data/sub_train_f.csv"

## label encoder for gbdt input
## one-hot encoder for gbdt output
fp_lb_enc = "../data/lb_enc"
fp_oh_enc_gbdt = "../data/gbdt-lr/oh_enc_gbdt"

## pre-trained model storing
fp_lr_model = "../data/gbdt-lr/lr_model"
fp_gbdt_model = "../data/gbdt-lr/gbdt_model"

## submission files
fp_sub_gbdt = "../data/gbdt-lr/GBDT_submission.csv"
fp_sub_gbdt_lr = "../data/gbdt-lr/GBDT-LR_submission.csv"

##==================== GBDT-LR training ====================##
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

#----- data for GBDT (label encoding) -----#
df_train = pd.read_csv(fp_sub_train_f)  # data load

## label transform for training set
label_enc = pickle.load(open(fp_lb_enc, 'rb'))
for col in cols:
    df_train[col] = label_enc[col].transform(df_train[col].values)

## split the data for the training of GBDT and LR respectively 
## to avoid over-fitting
X_train_org = df_train[cols].get_values()
y_train_org = df_train['click'].get_values()
X_train, X_valid, y_train, y_valid = train_test_split(X_train_org, y_train_org, test_size = 0.3, random_state = 0)
X_train_gbdt, X_train_lr, y_train_gbdt, y_train_lr = train_test_split(X_train, y_train, test_size = 0.5, random_state = 0)

del df_train
del X_train
del y_train
gc.collect()

#----- GBDT training -----#
#----- parameters tuning of GBDT -----#

param = {  # init the hyperparams of GBDT
    'learning_rate': 0.2,
    'n_estimators': 100,  # number of trees here
    'max_depth': 8,  # set max_depth of a tree
    'min_samples_split': 20, 
    'min_samples_leaf': 10,
    'subsample': 0.01, 
    'max_leaf_nodes': None,  # set max leaf nodes of a tree
    'random_state': 1,
    'verbose': 0
    }

gbdt_model = GradientBoostingClassifier()
gbdt_model.set_params(**param)

'''
#----- parameters tuning of GBDT -----#
### n_estimators
log_loss_train = []
log_loss_valid = []
n_estimators = [10,20,30,40,50,60,70,80,90,100,120,140]
for nt in n_estimators:
    print('training: n_estimators = ', nt)
    
    param['n_estimators'] = nt
    gbdt_model.set_params(**param)
    gbdt_model.fit(X_train_gbdt, y_train_gbdt)

    # scores
    y_pred_gbdt = gbdt_model.predict_proba(X_train_gbdt)[:, 1]
    log_loss_gbdt = log_loss(y_train_gbdt, y_pred_gbdt)
    print('log loss of GBDT on train set: %.5f' % log_loss_gbdt)
    log_loss_train.append(log_loss_gbdt)
    
    y_pred_gbdt = gbdt_model.predict_proba(X_valid)[:, 1]
    log_loss_gbdt = log_loss(y_valid, y_pred_gbdt)
    print('log loss of GBDT on valid set: %.5f' % log_loss_gbdt)
    log_loss_valid.append(log_loss_gbdt)
    
## plot the curve
f1 = plt.figure(1)
plt.plot(n_estimators, log_loss_train, label='train')
plt.plot(n_estimators, log_loss_valid, label='valid')
plt.xlabel('n_estimators')
plt.ylabel('log_loss')
plt.title('n_estimators (md=8,lr=0.2,mss=2,msl=1)')
plt.legend()
plt.grid(True, linewidth=0.3)
plt.show()

param['n_estimators'] = 100

### max_depth
log_loss_train = []
log_loss_valid = []
max_depths = [4,5,6,7,8,9,10,11,12]
for md in max_depths:
    print('training: max_depth = ', md)
    
    param['max_depth'] = md
    gbdt_model.set_params(**param)
    gbdt_model.fit(X_train_gbdt, y_train_gbdt)

    # scores
    y_pred_gbdt = gbdt_model.predict_proba(X_train_gbdt)[:, 1]
    log_loss_gbdt = log_loss(y_train_gbdt, y_pred_gbdt)
    print('log loss of GBDT on train set: %.5f' % log_loss_gbdt)
    log_loss_train.append(log_loss_gbdt)
    
    y_pred_gbdt = gbdt_model.predict_proba(X_valid)[:, 1]
    log_loss_gbdt = log_loss(y_valid, y_pred_gbdt)
    print('log loss of GBDT on valid set: %.5f' % log_loss_gbdt)
    log_loss_valid.append(log_loss_gbdt)
    
## plot the curve
f1 = plt.figure(2)
plt.plot(max_depths, log_loss_train, label='train')
plt.plot(max_depths, log_loss_valid, label='valid')
plt.xlabel('max_depth')
plt.ylabel('log_loss')
plt.title('max_depth (nt=100,lr=0.2,mss=20,msl=10)')
plt.legend()
plt.grid(True, linewidth=0.3)
plt.show()

param['max_depth'] = 9

### min_samples_split
log_loss_train = []
log_loss_valid = []
min_samples_splits = [2,5,10,15,20,25,30,35,40,50,60,70,80]
for mss in min_samples_splits:
    print('training: min_samples_split = ', mss)
    
    param['min_samples_split'] = mss
    gbdt_model.set_params(**param)
    gbdt_model.fit(X_train_gbdt, y_train_gbdt)

    # scores
    y_pred_gbdt = gbdt_model.predict_proba(X_train_gbdt)[:, 1]
    log_loss_gbdt = log_loss(y_train_gbdt, y_pred_gbdt)
    print('log loss of GBDT on train set: %.5f' % log_loss_gbdt)
    log_loss_train.append(log_loss_gbdt)
    
    y_pred_gbdt = gbdt_model.predict_proba(X_valid)[:, 1]
    log_loss_gbdt = log_loss(y_valid, y_pred_gbdt)
    print('log loss of GBDT on valid set: %.5f' % log_loss_gbdt)
    log_loss_valid.append(log_loss_gbdt)
    
## plot the curve
f1 = plt.figure(3)
plt.plot(min_samples_splits, log_loss_train, label='train')
plt.plot(min_samples_splits, log_loss_valid, label='valid')
plt.xlabel('min_samples_split')
plt.ylabel('log_loss')
plt.title('min_samples_split (nt=50,lr=0.2,md=8,msl=1)')
plt.legend()
plt.grid(True, linewidth=0.3)
plt.show()

param['min_samples_split'] = 40

### min_samples_leaf
log_loss_train = []
log_loss_valid = []
min_samples_leafs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,20,23,26,30]
for msl in min_samples_leafs:
    print('training: min_samples_leaf = ', msl)
    
    param['min_samples_leaf'] = msl
    gbdt_model.set_params(**param)
    gbdt_model.fit(X_train_gbdt, y_train_gbdt)

    # scores
    y_pred_gbdt = gbdt_model.predict_proba(X_train_gbdt)[:, 1]
    log_loss_gbdt = log_loss(y_train_gbdt, y_pred_gbdt)
    print('log loss of GBDT on train set: %.5f' % log_loss_gbdt)
    log_loss_train.append(log_loss_gbdt)
    
    y_pred_gbdt = gbdt_model.predict_proba(X_valid)[:, 1]
    log_loss_gbdt = log_loss(y_valid, y_pred_gbdt)
    print('log loss of GBDT on valid set: %.5f' % log_loss_gbdt)
    log_loss_valid.append(log_loss_gbdt)
    
## plot the curve
f1 = plt.figure(4)
plt.plot(min_samples_leafs, log_loss_train, label='train')
plt.plot(min_samples_leafs, log_loss_valid, label='valid')
plt.xlabel('min_samples_leaf')
plt.ylabel('log_loss')
plt.title('min_samples_leaf (nt=25,lr=0.2,md=7,mss=35)')
plt.legend()
plt.grid(True, linewidth=0.3)
plt.show()

param['min_samples_leaf'] = 10
'''

''''
## fitting
gbdt_model.fit(X_train_gbdt, y_train_gbdt)

## log-loss of training
y_pred_gbdt = gbdt_model.predict_proba(X_train_gbdt)[:, 1]
log_loss_gbdt = log_loss(y_train_gbdt, y_pred_gbdt)
print('log loss of GBDT on train set: %.5f' % log_loss_gbdt)

y_pred_gbdt = gbdt_model.predict_proba(X_valid)[:, 1]
log_loss_gbdt = log_loss(y_valid, y_pred_gbdt)
print('log loss of GBDT on valid set: %.5f' % log_loss_gbdt)

## store the pre-trained gbdt_model
pickle.dump(gbdt_model, open(fp_gbdt_model, 'wb'))
'''

del X_train_gbdt
del y_train_gbdt
gc.collect()

gbdt_model = pickle.load(open(fp_gbdt_model, 'rb'))
#----- data for LR (one-hot encoding with GDBT output) -----#
id_cols = []
for i in range(1, gbdt_model.get_params()['n_estimators']+1):
    id_cols.append('tree'+str(i))
oh_enc = OneHotEncoder(id_cols)

'''
def chunker(seq, size):
    return (seq[pos: pos + size] for pos in range(0, len(seq), size))

## oh_enc fit the train_set
df_train_id = pd.DataFrame(gbdt_model.apply(X_train_org)[:, :, 0], columns=id_cols, dtype=np.int8)

for chunk in chunker(df_train_id, 50000):
    oh_enc.fit(chunk)
    
del df_train_id
'''
del X_train_org
del y_train_org
gc.collect()

'''
## oh_enc fit the test_set
df_test_f = pd.read_csv(fp_test_f, 
                        index_col=None,  dtype={'id':str}, 
                        chunksize=50000, iterator=True)

for chunk in df_test_f:
    ## label transform for training set
    for col in cols:
        chunk[col] = label_enc[col].transform(chunk[col].values)       
    X_test = chunk[cols].get_values()
    
    #----- GBDT-LR -----#
    df_X_test_id = pd.DataFrame(gbdt_model.apply(X_test)[:, :, 0], columns=id_cols, dtype=np.int8)  # gbdt
    oh_enc.fit(df_X_test_id)

## store the encoder
pickle.dump(oh_enc, open(fp_oh_enc_gbdt, 'wb'))
'''

oh_enc = pickle.load(open(fp_oh_enc_gbdt, 'rb'))

#---- LR model -----#
lr_model = SGDClassifier(loss='log')  # using log-loss for LogisticRegression

'''
## input data (one-hot encoding)
df_X_train_lr_id = pd.DataFrame(gbdt_model.apply(X_train_lr)[:, :, 0], columns=id_cols, dtype=np.int8)
df_X_train_lr_id['click'] = y_train_lr

## fitting
for chunk in chunker(df_X_train_lr_id,10000):
    X_train = oh_enc.transform(chunk[id_cols])
    y_train = chunk['click'].astype('int')
    lr_model.partial_fit(X_train, y_train, classes = [0,1]) 

## log-loss of training
log_loss_lr = []
for chunk in chunker(df_X_train_lr_id,10000):
    X_train_id = oh_enc.transform(chunk[id_cols])
    y_pred_lr = lr_model.predict_proba(X_train_id)[:, 1]
    log_loss_lr_tmp = log_loss(y_train_lr, y_pred_lr)
#     print('log loss of LR on train set: %.5f' % log_loss_lr_tmp)
    log_loss_lr.append(log_loss_lr_tmp)

del df_X_train_lr_id
gc.collect()

## log-loss of valid
log_loss_lr = []
df_X_valid_id = pd.DataFrame(gbdt_model.apply(X_valid)[:, :, 0], columns=id_cols, dtype=np.int8)

for chunk in chunker(df_X_train_lr_id,10000):
    X_train_id = oh_enc.transform(chunk[id_cols])
    y_pred_lr = lr_model.predict_proba(X_train_id)[:, 1]
    log_loss_lr_tmp = log_loss(y_train_lr, y_pred_lr)
#     print('log loss of LR on train set: %.5f' % log_loss_lr_tmp)
    log_loss_lr.append(log_loss_lr_tmp)
    
X2_valid = oh_enc.transform(gbdt_model.apply(X_valid)[:, :, 0])
y_pred_lr = lr_model.predict_proba(X2_valid)[:, 1]
log_loss_lr = log_loss(y_valid, y_pred_lr)
print('log loss of LR on valid set: %.5f' % log_loss_lr)

## store the pre-trained model
pickle.dump(lr_model, open(fp_lr_model, 'wb'))
'''

lr_model = pickle.load(open(fp_lr_model, 'rb'))
##==================== Prediction ====================##
df_test_f = pd.read_csv(fp_test_f, 
                        index_col=None,  dtype={'id':str}, 
                        chunksize=1000, iterator=True)        

hd = False
for chunk in df_test_f:
    ## label transform for training set
    for col in cols:
        chunk[col] = label_enc[col].transform(chunk[col].values)       
    X_test = chunk[cols].get_values()
    
    #----- GBDT-LR -----#
    y_pred_gbdt = gbdt_model.predict_proba(X_test)[:, 1]
    X_test_gbdt = pd.DataFrame(gbdt_model.apply(X_test)[:, :, 0], columns=id_cols, dtype=np.int8)
    X2_test = oh_enc.transform(X_test_gbdt)  # one-hot
    y_pred_lr = lr_model.predict_proba(X2_test)[:, 1]  # lr   
    
    #----- generation of submission -----#
    chunk['click_gbdt'] = y_pred_gbdt
    chunk['click_lr'] = y_pred_lr
    with open(fp_sub_gbdt, 'a') as f: 
        chunk.to_csv(f, columns=['id', 'click_gbdt'], header=hd, index=False)
    with open(fp_sub_gbdt_lr, 'a') as f: 
        chunk.to_csv(f, columns=['id', 'click_lr'], header=hd, index=False)
    hd = False
    
print(' - PY131 - ')