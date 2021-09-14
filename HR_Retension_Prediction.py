#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the library
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle


# In[2]:


df = pd.read_csv("HR_comma_sep.csv")
# Show first five rows from data set
df.head()


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


df.dtypes


# In[6]:


df.describe()


# In[7]:


df['left'].value_counts()


# In[8]:


#Creating dummies for salary
s_dummies=pd.get_dummies(df["salary"])
s_dummies.head()


# In[9]:


df["s_high"]=s_dummies["high"]
df["s_low"]=s_dummies["low"]


# In[10]:


df.head()


# In[11]:


df=df.drop(["salary","Department"],1)


# In[12]:


df.head()


# In[13]:


df.dtypes


# In[14]:


#Split the dataset into training and test dataset
df_train, df_test = train_test_split(df,test_size=0.2,random_state=1)


# In[15]:


print('Train -',df_train.shape)
print('Test -',df_test.shape)


# In[16]:


X_train=df_train.drop(["left"],1)
Y_train=df_train["left"]
X_test=df_test.drop(["left"],1)
Y_test=df_test["left"]


# # Gradient Boosting Classifier

# In[40]:


#import Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier


# In[41]:


gbm_model=GradientBoostingClassifier()
gbm_model.fit(X_train,Y_train)
gbmPredict=gbm_model.predict(X_test)


# In[42]:


print("Accuracy:",metrics.accuracy_score(Y_test,gbmPredict))
print(confusion_matrix(Y_test,gbmPredict))


# In[43]:


print('Accuracy:',r2_score(Y_test,gbmPredict))


# In[44]:


print(classification_report(Y_test,gbmPredict))


# In[45]:


parameters = {
    "learning_rate": [0.01, 0.1, 0.15],
    "max_depth":[1,2,3,5],
    "subsample": [0.4,0.6,0.9],
    "n_estimators":[100,200,400]
    }


# In[46]:


Grad_clf = GridSearchCV(gbm_model, parameters, verbose=1, n_jobs=3, cv=5)
Grad_clf.fit(X_train, Y_train)


# In[47]:


Tuned_Grad=Grad_clf.best_estimator_
print(Tuned_Grad)


# In[48]:


Grad_predicted=Tuned_Grad.predict(X_test)


# In[49]:


print('Accuracy:',r2_score(Y_test,Grad_predicted))


# In[50]:


Grad_clf.score(X_test, Y_test)

pickle.dump(Grad_clf,open("HR_Retension_Prediction.pkl","wb"))


