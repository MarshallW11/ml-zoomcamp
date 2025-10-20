import pandas as pd
import numpy as np
import sklearn
import pickle


# In[34]:


print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')


# In[35]:


from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

customer = {
    'gender': 'male',
    'seniorcitizen': 0,
    'partner': 'no',
    'dependents': 'yes',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 6,
    'monthlycharges': 29.85,
    'totalcharges': 129.85
}

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


churn = pipeline.predict_proba(customer)[0, 1]

if churn >= 0.5:
    print(f'Send promotional email {churn}')
else:
    print(f"Don't send promotional email {1 - churn}")
