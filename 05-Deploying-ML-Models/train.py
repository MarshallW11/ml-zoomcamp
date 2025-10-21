#!/usr/bin/env python
# coding: utf-8

# This is a starter notebook for an updated module 5 of ML Zoomcamp
# 
# The code is based on the modules 3 and 4. We use the same dataset: [telco customer churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)


import pandas as pd
import numpy as np
import sklearn
import pickle
from sklearn.pipeline import make_pipeline


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# In[104]:


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


def load_data():

    data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

    df = pd.read_csv(data_url)

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')

    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)

    df.churn = (df.churn == 'yes').astype(int)
    
    return df


def train_model(df):
    
    numerical = ['tenure', 'monthlycharges', 'totalcharges']

    categorical = [
        'gender',
        'seniorcitizen',
        'partner',
        'dependents',
        'phoneservice',
        'multiplelines',
        'internetservice',
        'onlinesecurity',
        'onlinebackup',
        'deviceprotection',
        'techsupport',
        'streamingtv',
        'streamingmovies',
        'contract',
        'paperlessbilling',]
    
    
    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        LogisticRegression(solver='liblinear')  
    )

    train_dict = df[categorical + numerical].to_dict(orient='records')
    
    y_train = df.churn.values


    pipeline.fit(train_dict, y_train)
    
    return pipeline


def save_model(output_file, model):
    with open(output_file, 'wb') as f_out:
        pickle.dump(model, f_out)
        
df = load_data()
pipeline = train_model(df)
save_model('model.bin', pipeline)


