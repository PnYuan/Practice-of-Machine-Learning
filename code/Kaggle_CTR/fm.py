"""
a demo of FM/FFM-based Avazu-CTR prediction.
    
    1.generate data set with the form of fm/ffm input;
    2.training and tuning fm/ffm;
    3.predicting to get submissions;
    
"""

##================ packages ==================##
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import gc
import pickle

import xlearn as xl

##================ file paths (fp) ==================##
## raw data 2 (for read)
fp_raw_train = "../data/sub_train_f.csv"
fp_raw_test  = "../data/test_f.csv"

## label encoder for features
fp_lb_enc = "../data/label_enc"

## input
fp_train = "../data/fm/train_ffm.txt"
fp_valid = "../data/fm/valid_ffm.txt"
fp_test  = "../data/fm/test_ffm.txt"

## output
fp_model_fm  = "../data/fm/model_fm.out"
fp_model_ffm = "../data/fm/model_ffm.out"
fp_pred_fm   = "../data/fm/output_fm.txt"
fp_pred_ffm  = "../data/fm/output_ffm.txt"

## submissions
fp_sub_fm  = "../data/fm/Submission_FM.csv"
fp_sub_ffm = "../data/fm/Submission_FFM.csv"

##================ data prepare ==================##
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

#----- label transform for data set -----#
label_enc = pickle.load(open(fp_lb_enc, 'rb'))

## train set
df_train = pd.read_csv(fp_raw_train)
for col in cols:
    df_train[col] = label_enc[col].transform(df_train[col].values)

## test set
df_test = pd.read_csv(fp_raw_test)
for col in cols:
    df_test[col] = label_enc[col].transform(df_test[col].values)
df_test['click'] = -1  # add a placeholder

##----- merge train-test set -----##
n_train = len(df_train)
n_test = len(df_test)
df = df_train.append(df_test)   
del df_train, df_test
gc.collect()


##----- format data file (format as libffm) for train/valid/test -----##
def convert_to_ffm(df, numerics, categories, features, Label, n_train, train_size=0.5):
    """
    :function: generation of train/valid/test set format as libffm 
    
    :parameters:
        :df, pandas dataframe include raw data of train and test.
        :numerics, name list of numerical features.
        :categories, name list of categorical features.
        :features, name list of all features.
        :Label, name of label in the df.
        :n_train, number of training samples.
        :train_size, the ratio of train_valid split.
    """
    catdict = {}
    # Flagging categorical and numerical fields
    for x in numerics:
        catdict[x] = 0
    for x in categories:
        catdict[x] = 1
    
    nrows = df.shape[0]
    
    # samples' number of train
    n1 = n_train * train_size
    
    with open(fp_train, "w") as file_train, \
         open(fp_valid, "w") as file_valid, \
         open(fp_test, "w")  as file_test:
        
        # Looping over rows to convert each row to libffm format
        for n, r in enumerate(range(nrows)):
            
            datastring = ""
            datarow = df.iloc[r].to_dict()
            datastring += str(int(datarow[Label]))
            # For  fields, we are creating a dummy field here
            for i, x in enumerate(catdict.keys()):
                if(catdict[x]==0):  # numerical
                    datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
                else:  # categorical
                    datastring = datastring + " "+str(i)+":"+ str(int(datarow[x]))+":1"
            datastring += '\n'
            
            if n < n1: file_train.write(datastring)
            elif n < n_train: file_valid.write(datastring)
            else: file_test.write(datastring)

convert_to_ffm(df, numerics=[], categories=cols, features=cols, Label='click', n_train=n_train, train_size=0.8)

##================ FM ==================##
## setting
fm_model = xl.create_fm()  # Use factorization machine
fm_model.setTrain(fp_train)   # Training data
fm_model.setValidate(fp_valid)  # Validation data
fm_model.setSigmoid()

param = {'task': 'binary',
         'k': 20,
         'lr': 0.02, 
         'lambda': 0.002,
         'epoch': 100,
         'opt': 'adagrad'
         }

## training
fm_model.fit(param, fp_model_fm)

## testing
fm_model.setTest(fp_test)
fm_model.setSigmoid()
fm_model.predict(fp_model_fm, fp_pred_fm)

##================ FFM ==================##
## training setting
ffm_model = xl.create_ffm()  # Use field-aware factorization machine
ffm_model.setTrain(fp_train)   # Training data
ffm_model.setValidate(fp_valid)  # Validation data
ffm_model.setSigmoid()

param = {'task': 'binary',
         'k': 20,
         'lr': 0.02, 
         'lambda': 0.0001,
         'epoch': 100,
         'opt': 'adagrad'
         }

## Train model
ffm_model.fit(param, fp_model_ffm)

## Test model
ffm_model.setTest(fp_test)
ffm_model.setSigmoid()
ffm_model.predict(fp_model_ffm, fp_pred_ffm)

##================ Get Submission ==================##
#----- fm -----#
y_pred_fm = np.loadtxt(fp_pred_fm)
df_test = pd.read_csv(fp_raw_test, dtype={'id':str})
df_test['click'] = y_pred_fm

with open(fp_sub_fm, 'w') as f: 
    df_test.to_csv(f, columns=['id', 'click'], header=True, index=False)

#----- ffm -----#
y_pred_ffm = np.loadtxt(fp_pred_ffm)
df_test = pd.read_csv(fp_raw_test, dtype={'id':str})
df_test['click'] = y_pred_ffm

with open(fp_sub_ffm, 'w') as f: 
    df_test.to_csv(f, columns=['id', 'click'], header=True, index=False)

print(' - PY131 - ')