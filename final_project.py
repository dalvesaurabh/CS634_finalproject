#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


os.listdir('breast-cancer-detection')


# In[3]:


data = pd.read_csv('breast-cancer-detection/train.csv')
data.columns


# In[4]:


data = data.drop(['Unnamed: 32','id'], axis = 1)
data.head()


# In[5]:


for col in data.columns:
    if data[col].dtypes == 'object':
        le = preprocessing.LabelEncoder()
        data[col] = le.fit_transform(data[col])
    else:
        scaler = preprocessing.StandardScaler()
        data[col] = scaler.fit_transform(np.array(data[col]).reshape(-1, 1))


# In[6]:


data['diagnosis'].hist()


# In[7]:


for column in data.columns:
    plt.figure()
    sns.distplot(data[column])


# In[8]:


data.shape


# In[9]:


X, y = np.array(data[data.columns[:-1]]), np.array(data[data.columns[-1]])


# In[10]:


def metric(tn, fp, fn, tp):
    result = []
    tpr = tp/(tp+fn)
    result.append(tpr)
    tnr = tn/(tn+fp)
    result.append(tnr)
    fpr = fp/(tn+fp)
    result.append(fpr)
    fnr = fn/(tp+fn)
    result.append(fnr)
    recall = tp/(tp+fn)
    result.append(recall)
    precision = tp/(tp+fp)
    result.append(precision)
    f1 = (2*tp)/(2*tp+fp+fn)
    result.append(f1)
    acc = (tp+tn)/(tp+fp+fn+tn)
    result.append(acc)
    err = (fp+fn)/(tp+fp+fn+tn)
    result.append(err)
    bacc = (tpr+tnr)/2
    result.append(bacc)
    tss = tp/(tp+fn) - fp/(fp+tn)
    result.append(tss)
    hss = 2*(tp*tn - fp*fn)/((tp+fn)*(fn+tn) + (tp+fp)*(fp+tn))
    result.append(hss)
    return np.array(result)


# In[11]:


kf = KFold(n_splits=10, random_state=123)
fold = 0
svc_mean = np.zeros(12)
rf_mean = np.zeros(12)
gnb_mean = np.zeros(12)
for train_index, test_index in kf.split(X, y):
    fold += 1
    print("Fold", str(fold))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #SVM
    print("\tSVM model result:")
    svc = SVC(gamma='auto')
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_svc).ravel()
    svc_result = metric(tn, fp, fn, tp)
    svc_mean += svc_result
    print("\t\tTrue positive rate:", svc_result[0])
    print("\t\tTrue negative rate:", svc_result[1])
    print("\t\tFalse positive rate:", svc_result[2])
    print("\t\tFalse negative rate:", svc_result[3])
    print("\t\tRecall:", svc_result[4])
    print("\t\tPrecision:", svc_result[5])
    print("\t\tF1:", svc_result[6])
    print("\t\tAccuracy:", svc_result[7])
    print("\t\tError Rate:", svc_result[8]) 
    print("\t\tBalance Accuracy:", svc_result[9])
    print("\t\tTrue skill statistics:", svc_result[10])
    print("\t\tHeidke skill score:", svc_result[11])
    #Random Forrest
    print("\tRandom Forest model result:")
    rf = RandomForestClassifier(max_depth=5, random_state=0)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()
    rf_result = metric(tn, fp, fn, tp)
    rf_mean += rf_result
    print("\t\tTrue positive rate:", rf_result[0])
    print("\t\tTrue negative rate:", rf_result[1])
    print("\t\tFalse positive rate:", rf_result[2])
    print("\t\tFalse negative rate:", rf_result[3])
    print("\t\tRecall:", rf_result[4])
    print("\t\tPrecision:", rf_result[5])
    print("\t\tF1:", rf_result[6])
    print("\t\tAccuracy:", rf_result[7])
    print("\t\tError Rate:", rf_result[8]) 
    print("\t\tBalance Accuracy:", rf_result[9])
    print("\t\tTrue skill statistics:", rf_result[10])
    print("\t\tHeidke skill score:", rf_result[11])
    #Naive Bayes
    print("\tNaive Bayes model result:")
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_gnb).ravel()
    gnb_result = metric(tn, fp, fn, tp)
    gnb_mean += gnb_result
    print("\t\tTrue positive rate:", gnb_result[0])
    print("\t\tTrue negative rate:", gnb_result[1])
    print("\t\tFalse positive rate:", gnb_result[2])
    print("\t\tFalse negative rate:", gnb_result[3])
    print("\t\tRecall:", gnb_result[4])
    print("\t\tPrecision:", gnb_result[5])
    print("\t\tF1:", gnb_result[6])
    print("\t\tAccuracy:", gnb_result[7])
    print("\t\tError Rate:", gnb_result[8]) 
    print("\t\tBalance Accuracy:", gnb_result[9])
    print("\t\tTrue skill statistics:", gnb_result[10])
    print("\t\tHeidke skill score:", gnb_result[11])
#     print(y_pred_svc.shape, y_pred_rf.shape, y_pred_gnb.shape)


# In[12]:


svc_mean /= 10
rf_mean /= 10
gnb_mean /= 10


# In[13]:


print("Overall result for SVM model:")
print("\tTrue positive rate:", svc_mean[0])
print("\tTrue negative rate:", svc_mean[1])
print("\tFalse positive rate:", svc_mean[2])
print("\tFalse negative rate:", svc_mean[3])
print("\tRecall:", svc_mean[4])
print("\tPrecision:", svc_mean[5])
print("\tF1:", svc_mean[6])
print("\tAccuracy:", svc_mean[7])
print("\tError Rate:", svc_mean[8]) 
print("\tBalance Accuracy:", svc_mean[9])
print("\tTrue skill statistics:", svc_mean[10])
print("\tHeidke skill score:", svc_mean[11])


# In[14]:


print("Overall result for Random Forest model:")
print("\tTrue positive rate:", rf_mean[0])
print("\tTrue negative rate:", rf_mean[1])
print("\tFalse positive rate:", rf_mean[2])
print("\tFalse negative rate:", rf_mean[3])
print("\tRecall:", rf_mean[4])
print("\tPrecision:", rf_mean[5])
print("\tF1:", rf_mean[6])
print("\tAccuracy:", rf_mean[7])
print("\tError Rate:", rf_mean[8]) 
print("\tBalance Accuracy:", rf_mean[9])
print("\tTrue skill statistics:", rf_mean[10])
print("\tHeidke skill score:", rf_mean[11])


# In[15]:


print("Overall result for Naive Bayes model:")
print("\tTrue positive rate:", gnb_mean[0])
print("\tTrue negative rate:", gnb_mean[1])
print("\tFalse positive rate:", gnb_mean[2])
print("\tFalse negative rate:", gnb_mean[3])
print("\tRecall:", gnb_mean[4])
print("\tPrecision:", gnb_mean[5])
print("\tF1:", gnb_mean[6])
print("\tAccuracy:", gnb_mean[7])
print("\tError Rate:", gnb_mean[8]) 
print("\tBalance Accuracy:", gnb_mean[9])
print("\tTrue skill statistics:", gnb_mean[10])
print("\tHeidke skill score:", gnb_mean[11])


# In[ ]:




