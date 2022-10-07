#!/usr/bin/env python
# coding: utf-8


# In[1]:


get_ipython().system('pip install imbalanced-learn')


# In[2]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import xgboost as xg
from sklearn.model_selection import GridSearchCV


# ## (b) Data Preparation
# ## This data set has missing values. When the number of data with missing values is significant, discarding them is not a good idea.

# ### i. Research what types of techniques are usually used for dealing with data with missing values. Pick at least one of them and apply it to this data in the next steps.

# #### It is usually to replace the missing values with the mean or median of the column. I will do the 'mean' one.

# In[3]:


test_set = pd.read_csv('../data/aps_failure_test_set.csv', skiprows = range (0,20))
training_set = pd.read_csv('../data/aps_failure_training_set.csv',skiprows = range (0,20))
training_set


# In[4]:


full_set= pd.concat([training_set,test_set])
columns_lst=full_set.columns.values.tolist()
full_set


# In[5]:


y_full=full_set.iloc[:,0:1]
y_full.reset_index(drop=True,inplace=True)

trans_set = full_set.iloc[:,1:].replace('na',np.nan)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(trans_set)
trans_set = imp.transform(trans_set)
trans_set = pd.DataFrame(trans_set)

full_set=pd.concat([y_full,trans_set],axis=1)
full_set.columns = columns_lst

full_set


# ### ii. For each of the 170 features, calculate the coefficient of variation CV = s/m, where s is sample standard deviation and m is sample mean.

# In[6]:


CV_dict_train = {}
for i in columns_lst[1:]:
    s = statistics.stdev(full_set[i])
    m = statistics.mean(full_set[i])
    CV = s/m
    CV_dict_train[i]= CV
    print('The Coefficient of Variation for', i ,'is:', CV)


# ### iii. Plot a correlation matrix for your features using pandas or any other tool.

# In[7]:


cor_mat = full_set.iloc[:,1:].corr()
round(cor_mat,2)


# In[8]:


fig = plt.figure(figsize=(45,45))
ax = fig.add_subplot()
ax = sns.heatmap(cor_mat, linewidths=0.05,vmax=1, vmin=0 ,annot=True,annot_kws={'size':6,'weight':'bold'})
ax.set_title('Correlation matrix')
#plt.savefig('cor_mat1.tif',dpi=1000)
plt.show()


# ### iv. Pick √170 features with highest CV , and make scatter plots and box plots for them, similar to those on p.  129 of ISLR. Can you draw conclusions about significance of those features, just by the scatter plots? This does not mean that you will only use those features in the following questions. We picked them only for visualization.

# In[9]:


N = int(math.sqrt(170))
N


# In[10]:


ranking = sorted(CV_dict_train.items(), key=lambda x: x[1], reverse=True)
print(ranking[:13])


# In[11]:


chosen_lst = ['cf_000','co_000','ad_000','cs_009','dj_000','as_000','dh_000','df_000','ag_000','au_000','ak_000','az_009','ay_009']
chosen_table = full_set.loc[:,chosen_lst]
graph = sns.PairGrid(chosen_table) 
graph = graph.map_diag(plt.hist) 
graph = graph.map_offdiag(plt.scatter)
plt.show()


# In[12]:


#sns.pairplot(full_set.loc[:,chosen_lst])


# ### v. Determine the number of positive and negative data. Is this data set imbalanced?

# In[13]:


label=full_set['class'].tolist()
data_dict = {}
for d in label:
    if d not in data_dict:
        data_dict[d]=0
    data_dict[d]= data_dict[d]+1
print(data_dict)   


# In[14]:


total = 0
for val in data_dict.values():
    total += val

for key,val in data_dict.items():
    ratio = float(val/total)
    data_dict[key] = ratio
    
print(data_dict)


# #### So the ratio of negative data is 98% while positive data is 1.8%. So this data set is imbalanced.

# ### (c) Train a random forest to classify the data set. Do NOT compensate for class imbalance in the data set. Calculate the confusion matrix, ROC, AUC, and misclassification for training and test sets and report them (You may use pROC package). Calculate Out of Bag error estimate for your random forset and compare it to the test error.

# In[15]:


full_set['class'] = full_set['class'].replace({'neg': 0, 'pos': 1}).astype(int)
training = full_set.iloc[:60000,:]
test = full_set.iloc[60000:,:]
X_train = training.iloc[:,1:]
y_train = training.iloc[:,0:1]
X_test = test.iloc[:,1:]
y_test = test.iloc[:,0:1]

clf = RandomForestClassifier()
clf.fit(X_train,y_train.values.ravel())
y_pred =clf.predict(X_test)
y_pred_train = clf.predict(X_train)
print(classification_report(y_test, y_pred))


# In[16]:


accura = accuracy_score(y_pred, y_test)
test_error = 1 - accura
accura2 = accuracy_score(y_pred_train, y_train)
train_error = 1 - accura2
print('misclassification for test set is',test_error)
print('misclassification for train set is',train_error)


# In[17]:


cmf=confusion_matrix(y_test, y_pred)
print(cmf)


# In[18]:


y_pred_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)


# In[19]:


metrics.auc(fpr, tpr)


# In[20]:


forest = RandomForestClassifier(bootstrap=True, oob_score = True)
forest.fit(X_train, y_train.values.ravel())
print('Out of Bag error estimate: ', forest.score(X_test, y_test))


# #### So out of bag error is much higher than test error.

# ### (d) Research how class imbalance is addressed in random forests. Compensate for class imbalance in your random forest and repeat 1c. Compare the results with those of 1c.

# In[21]:


oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X_train, y_train)


# In[22]:


clf_2 = RandomForestClassifier(class_weight='balanced')
clf_2.fit(X_over,y_over.values.ravel())
y_pred_2 =clf.predict(X_test)
y_pred_train_2 = clf.predict(X_over)
print(classification_report(y_test, y_pred_2))


# In[23]:


accura_2 = accuracy_score(y_pred_2, y_test)
test_error_2 = 1 - accura_2
accura2_2 = accuracy_score(y_pred_train_2, y_over)
train_error_2 = 1 - accura2_2
print('misclassification for test set is',test_error_2)
print('misclassification for train set is',train_error_2)


# In[24]:


cmf_2=confusion_matrix(y_test, y_pred_2)
print(cmf_2)


# In[25]:


y_pred_prob_2 = clf_2.predict_proba(X_test)[:, 1]
fpr2, tpr2, thresholds2 = metrics.roc_curve(y_test, y_pred_prob_2)
plt.plot(fpr2, tpr2)


# In[26]:


metrics.auc(fpr2, tpr2)


# ### (e) XGBoost and Model Trees

# In[27]:


#le = LabelEncoder()
#y_train = le.fit_transform(y_train)
parameters = {'alpha':np.arange(0,1.1,0.1)}
estimator = xg.XGBClassifier(objective= 'reg:logistic')
grid_search = GridSearchCV(estimator=estimator,param_grid=parameters,scoring = 'accuracy',cv=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)


# In[28]:


print(grid_search.best_score_)


# In[29]:


grid_search.best_estimator_.fit(X_train, y_train)
y_predict=grid_search.best_estimator_.predict(X_test)
print('The α is 0.2\n','Test Error is:', metrics.accuracy_score(y_test,y_predict))


# #### So the error of your trained model is a little bit higher than the test error.

# In[30]:


cmf_3=confusion_matrix(y_test, y_predict)
print(cmf_3)


# In[31]:


y_pred_prob_3 = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
fpr3, tpr3, thresholds3 = metrics.roc_curve(y_test, y_pred_prob_3)
plt.plot(fpr3, tpr3)


# In[32]:


metrics.auc(fpr3, tpr3)


# ### (f) Use SMOTE (Synthetic Minority Over-sampling Technique) to pre-process your data to compensate for class imbalance.4 Train XGBosst with L1-penalized logistic regression at each node using the pre-processed data and repeat 1e. Do not forget that there is a right and a wrong way of cross validation here. Compare the uncompensated case with SMOTE case.

# In[33]:


smote = SMOTE()
X_sm, y_sm = smote.fit_resample(X_train, y_train)


# In[34]:


parameters = {'alpha':np.arange(0,1.1,0.1)}
estimator2 = xg.XGBClassifier(objective= 'reg:logistic')
grid_search2 = GridSearchCV(estimator=estimator2,param_grid=parameters,scoring = 'accuracy',cv=5)
grid_search2.fit(X_sm, y_sm)
print(grid_search2.best_estimator_)


# In[35]:


print(grid_search2.best_score_)


# In[36]:


grid_search2.best_estimator_.fit(X_sm, y_sm)
y_predict2=grid_search2.best_estimator_.predict(X_test)
print('The α is 0.2\n','Test Error is:', metrics.accuracy_score(y_test,y_predict2))


# #### So the error of your trained model is a little bit higher than the test error.

# In[37]:


cmf_4=confusion_matrix(y_test, y_predict2)
print(cmf_4)


# In[38]:


y_pred_prob_4 = grid_search2.best_estimator_.predict_proba(X_test)[:, 1]
fpr4, tpr4, thresholds4 = metrics.roc_curve(y_test, y_pred_prob_4)
plt.plot(fpr4, tpr4)


# In[39]:


metrics.auc(fpr4, tpr4)


# #### Overall, the test error and auc score of SMOTE case is better.

# ### References:
# #### https://datascience.stackexchange.com/questions/13151/randomforestclassifier-oob-scoring-method
# #### https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# #### https://www.geeksforgeeks.org/imbalanced-learn-module-in-python/
# #### https://blog.csdn.net/weixin_35437039/article/details/113015865
# #### https://www.kaggle.com/code/rafjaa/resampling-strategies-for-imbalanced-datasets/notebook
# #### http://ethen8181.github.io/machine-learning/regularization/regularization.html

# In[ ]:




