#!/usr/bin/env python
# coding: utf-8

# # Data Analysis
# This is the main notebook performing all feature engineering, model selection, training, evaluation etc.  
# The different steps are:
#  - Step1 - import dependencies
#  - Step2 - load payloads into memory
#  - Step3A - Feature engineering custom features
#  - Step3B - Feature engineering bag-of-words
#  - Step3C - Feature space visualization
#  - Step4 - Model selection
#  - (Step4B - Load pre-trained classifiers)
#  - Step5 - Visualization
#  - Step6 - Website integration extract

# # Step1
# import dependencies

# In[5]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
# import seaborn
import string
# from IPython.print import print
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier


import sklearn.gaussian_process.kernels as kernels

# from sklearn.cross_validation import ShuffleSplit
# from sklearn.cross_validation import KFold
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import roc_curve
# from sklearn.metrics import roc_auc_score

from scipy.stats import expon


# # Step2
# load the payloads into memory

# In[6]:

path = os.getcwd()
payloads = pd.read_csv(path + "/data/payloads.csv",index_col='index')
# print(payloads.head())


# # Step3A - feature engineering custom features  
# We will create our own feature space with features that might be important for this task, this includes:
#  - length of payload
#  - number of non-printable characters in payload
#  - number of punctuation characters in payload
#  - the minimum byte value of payload
#  - the maximum byte value of payload
#  - the mean byte value of payload
#  - the standard deviation of payload byte values
#  - number of distinct bytes in payload
#  - number of SQL keywords in payload
#  - number of javascript keywords in payload

# In[7]:


def plot_feature_distribution(features):
# #     print('Properties of feature: ' + features.name)
# #     print(features.describe())
# #     f, ax = plt.subplots(1, figsize=(10, 6))
# #     ax.hist(features, bins=features.max()-features.min()+1, normed=1)
# #     ax.set_xlabel('value')
# #     ax.set_ylabel('fraction')
    
# # #     plt.show()
    a = 0
    return 


# In[8]:


def create_feature_length(payloads):
    '''
        Feature describing the lengh of the input
    '''
    
    
    payloads['length'] = [len(str(row)) for row in payloads['payload']]
    return payloads


# payloads = create_feature_length(payloads)
# print(payloads.head())


# plot_feature_distribution(payloads['length'])


# In[10]:


def create_feature_non_printable_characters(payloads):  
    '''
    Feature
    Number of non printable characthers within payload
    '''
    
    payloads['non-printable'] = [ len([1 for letter in str(row) if letter not in string.printable]) for row in payloads['payload']]
    return payloads
    

# create_feature_non_printable_characters(payloads)
# print(payloads.head())
    
# plot_feature_distribution(payloads['non-printable'])


# In[11]:


def create_feature_punctuation_characters(payloads):
    '''
    Feature
    Number of punctuation characthers within payload
    '''
    
    payloads['punctuation'] = [ len([1 for letter in str(row) if letter in string.punctuation]) for row in payloads['payload']]
    return payloads
    

# create_feature_punctuation_characters(payloads)
# print(payloads.head())
    
# plot_feature_distribution(payloads['punctuation'])


# In[12]:


def create_feature_min_byte_value(payloads):
    '''
    Feature
    Minimum byte value in payload
    '''
    
    payloads['min-byte'] = [ min(bytearray(str(row), 'utf8')) for row in payloads['payload']]
    return payloads

# create_feature_min_byte_value(payloads)
# print(payloads.head())

# plot_feature_distribution(payloads['min-byte'])


# In[13]:


def create_feature_max_byte_value(payloads):
    '''
    Feature
    Maximum byte value in payload
    '''
    
    payloads['max-byte'] = [ max(bytearray(str(row), 'utf8')) for row in payloads['payload']]
    return payloads

# create_feature_max_byte_value(payloads)
# print(payloads.head())

# plot_feature_distribution(payloads['max-byte'])


# In[14]:


def create_feature_mean_byte_value(payloads):
    '''
    Feature
    Maximum byte value in payload
    '''
    
    payloads['mean-byte'] = [ np.mean(bytearray(str(row), 'utf8')) for row in payloads['payload']]
    return payloads

# create_feature_mean_byte_value(payloads)
# print(payloads.head())

# plot_feature_distribution(payloads['mean-byte'].astype(int))


# In[15]:


def create_feature_std_byte_value(payloads):
    '''
    Feature
    Standard deviation byte value in payload
    '''
    
    payloads['std-byte'] = [ np.std(bytearray(str(row), 'utf8')) for row in payloads['payload']]
    return payloads

create_feature_std_byte_value(payloads)
# print(payloads.head())

# plot_feature_distribution(payloads['std-byte'].astype(int))


# In[ ]:


def create_feature_distinct_bytes(payloads):
    '''
    Feature
    Number of distinct bytes in payload
    '''
    
    payloads['distinct-bytes'] = [ len(list(set(bytearray(str(row), 'utf8')))) for row in payloads['payload']]
    return payloads

# create_feature_distinct_bytes(payloads)
# print(payloads.head())

# plot_feature_distribution(payloads['distinct-bytes'])


# In[ ]:


sql_keywords = pd.read_csv(path + '/data/SQLKeywords.txt', index_col=False)

def create_feature_sql_keywords(payloads):
    
    '''
    Feature
    Number of SQL keywords within payload
    '''
    payloads['sql-keywords'] = [ len([1 for keyword in sql_keywords['Keyword'] if str(keyword).lower() in str(row).lower()]) for row in payloads['payload']]
    return payloads

# create_feature_sql_keywords(payloads)
# print(type(sql_keywords))
# print(payloads.head())
# plot_feature_distribution(payloads['sql-keywords'])
    


# In[ ]:


js_keywords = pd.read_csv( path +  '/data/JavascriptKeywords.txt', index_col=False)

def create_feature_javascript_keywords(payloads):
    '''
    Feature
    Number of Javascript keywords within payload
    '''
    
    payloads['js-keywords'] = [len([1 for keyword in js_keywords['Keyword'] if str(keyword).lower() in str(row).lower()]) for row in payloads['payload']]
    return payloads
    

# create_feature_javascript_keywords(payloads)
# print(payloads.head())    
# plot_feature_distribution(payloads['js-keywords'])
    
    


# define a function that makes a feature vector from the payload using the custom features

# In[ ]:


def create_features(payloads):
    print("Creating features ...")
    features = create_feature_length(payloads)
    features = create_feature_non_printable_characters(features)
    features = create_feature_punctuation_characters(features)
    features = create_feature_max_byte_value(features)
    features = create_feature_min_byte_value(features)
    features = create_feature_mean_byte_value(features)
    features = create_feature_std_byte_value(features)
    features = create_feature_distinct_bytes(features)
    features = create_feature_sql_keywords(features)
    features = create_feature_javascript_keywords(features)
    del features['payload']

    return features


# ### Scoring custom features
# Score the custom features using the SelectKBest function, then visualize the scores in a graph  
# to see which features are less significant

# In[15]:

def training():
    Y = payloads['is_malicious']
    X = create_features(pd.DataFrame(payloads['payload'].copy()))


    ########## For train a model we need to split the dataset into train and test######
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)


    ##########train SVM MODEL########
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.score(X_test,y_test)

    ###Saving the model
    pickle.dump( svclassifier, open( path + "/svm_model.p", "wb" ) )
    return
# training()
#model loading code
# classifier_results = pickle.load( open("C:/Users/ananthu.m1/Documents/works/kichu projct/svm_model.p", "rb" ) )