#!/usr/bin/env python
# coding: utf-8

# ## Loan Approval Prediction using Machine Learning
# #### Download the used data by visiting Kaggle.

# In[235]:


# Importing Libraries and Dataset
# Pandas – To load the Dataframe
# Matplotlib – To visualize the data features

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[236]:


# pip install seaborn


# In[237]:



train=pd.read_csv(r'C:\Users\Mathur\Desktop\AISHU\STUDY\Data Analytics\ML_Loan\train.csv')

train.Loan_Status=train.Loan_Status.map({'Y':1,'N':0})
train.head()
# data cleaning
train.isnull().sum()


# ### Data Preprocessing and Visualization

# In[238]:


obj = (train.dtypes == 'object') 
object_cols = list(obj[obj].index) 
plt.figure(figsize=(18,36)) 
index = 1

for col in object_cols: 
  y = train[col].value_counts() 
  plt.subplot(11,4,index) 
  plt.xticks(rotation=90) 
  sns.barplot(x=list(y.index), y=y) 
  index +=1


# ### Splitting Dataset into train & test

# In[239]:



# from sklearn.model_selection import train_test_split 
  
# X = data.drop(['Loan_Status'],axis=1) 
# Y = data['Loan_Status'] 
# X.shape,Y.shape 
  
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
#                                                     test_size=0.4, 
#                                                     random_state=1) 
# X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[240]:


train.shape


# In[241]:



test=pd.read_csv(r'C:\Users\Mathur\Desktop\AISHU\STUDY\Data Analytics\ML_Loan\test.csv')
train.info()


# In[242]:


# train.describe()
# train.columns
test.columns


# ## Data visualisation: Exploratory Data analysis

# In[243]:


train['Loan_Status'].value_counts().plot.bar()


# In[244]:


plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize = (10,8), title = 'Gender')
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(figsize = (10,8), title = 'Married')
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(figsize = (10,8), title = 'Self_Employed')
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(figsize = (10,8), title = 'Credit_History')
plt.show()


# In[245]:


plt.figure(1)
plt.subplot(231)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize = (10,8), title = 'Dependents')
plt.subplot(232)
train['Education'].value_counts(normalize=True).plot.bar(figsize = (10,8), title = 'Education')
plt.subplot(233)
train['Property_Area'].value_counts(normalize=True).plot.bar(figsize = (10,8), title = 'Property_Area')
plt.show()


# In[246]:


plt.figure(1)
plt.subplot(241)
sns.displot(train['ApplicantIncome'])
plt.subplot(242)
train['ApplicantIncome'].plot.box(figsize = (20,10))
plt.show()


# #### Co-applicant income visualisation

# In[247]:


plt.figure(1)
plt.subplot(251)
sns.histplot(train['CoapplicantIncome'])
plt.subplot(252)
train['CoapplicantIncome'].plot.box(figsize = (20,10))
plt.show()


# In[248]:


plt.figure(1)
plt.subplot(261)
sns.histplot(train['LoanAmount'])
plt.subplot(262)
train['LoanAmount'].plot.box(figsize = (20,10))
plt.show()


# In[249]:


# Get the number of columns of object datatype.

obj = (train.dtypes == 'object') 
print("Categorical variables:",len(list(obj[obj].index)))


# In[250]:


#categorical data: Loan_ID, Gender, Married, Dependents, Education, Self_Employed, Property_Area, Loan_Status
Gender = pd.crosstab(train['Gender'], train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (4,4))
plt.show()

Married = pd.crosstab(train['Married'], train['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (4,4))
plt.show()

Dependents = pd.crosstab(train['Dependents'], train['Loan_Status'])
Dependents.div(Dependents.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (4,4))
plt.show()

Education = pd.crosstab(train['Education'], train['Loan_Status'])
Education.div(Education.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (4,4))
plt.show()

Self_Employed = pd.crosstab(train['Self_Employed'], train['Loan_Status'])
Self_Employed.div(Self_Employed.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (4,4))
plt.show()

Property_Area = pd.crosstab(train['Property_Area'], train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (4,4))
plt.show()

Credit_History = pd.crosstab(train['Credit_History'], train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (4,4))
plt.show()


# ### Numerical Data
# 

# In[251]:


#numerical: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History
train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()


# ## Identifying patterns or trends in data using visualisation of [ correlation between features]

# In[252]:


bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High','Very high']
train['Income_bin'] = pd.cut(train['ApplicantIncome'],bins,labels = group)
Income_bin = pd.crosstab(train['Income_bin'], train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float),axis = 0).plot(kind ='bar',stacked = True) 
plt.xlabel('ApplicantIncome')
plt.ylabel('Percentage')
plt.show()


# In[253]:


bins=[0,1000,2000,41000] 
group=['Low','Average','High']
train['Coapplicant_Income_bin'] = pd.cut(train['CoapplicantIncome'],bins,labels = group)
Coapplicant_Income_bin = pd.crosstab(train['Coapplicant_Income_bin'], train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float),axis = 0).plot(kind ='bar',stacked = True) 
plt.xlabel('CoapplicantIncome')
plt.ylabel('Percentage')
plt.show()


# In[254]:


bins = [0,100,200,700]
group = ['Low','Average','High']
train['LoanAmount_bin'] = pd.cut(train['LoanAmount'],bins, labels = group)
LoanAmount_bin = pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis = 0).plot(kind ='bar',stacked = True)
plt.xlabel('LoanAmount')
plt.ylabel('Percentage')
plt.show()


# In[255]:


train['Total_Income'] = train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High','Very high']
train['Total_Income_bin'] = pd.cut(train['Total_Income'],bins,labels = group)
TotalIncome_bin = pd.crosstab(train['Total_Income_bin'], train['Loan_Status'])
TotalIncome_bin.div(TotalIncome_bin.sum(1).astype(float),axis = 0).plot(kind ='bar',stacked = True) 
plt.xlabel('Total_Income')
plt.ylabel('Percentage')
plt.show()


# In[257]:


train.head()


# In[258]:


train = train.drop(['Income_bin','Coapplicant_Income_bin','LoanAmount_bin','Total_Income','Total_Income_bin'], axis=1)


# In[259]:


# train.head(90)
# train['Loan_Status'].unique()
test.columns


# ## Data preprocessing/preparation: prepare the information for analysis

# In[260]:


# train['Dependents'].replace('3+',3,inplace = True)
# test['Dependents'].replace('3+',3,inplace = True)
# train['Loan_Status'].replace('N',0,inplace = True)
# train['Loan_Status'].replace('Y',1,inplace = True)


# In[261]:


# matrix = train.corr()
# f,ax = plt.subplots(figsize =(9,6))
# # sns.heatmap(matrix,vmax = .8,square= True, cmap='BuPu')
# plt.show()


# In[262]:


train.isnull().sum()


# ## Fixing data for missing values
# #### Categorical missing: Gender, Married, Dependents, Self_Employed, Property_Area, Loan_Status
# #### Numerical missing: LoanAmount, Loan_Amount_Term, Credit_History
# 

# In[265]:


train['Gender']= train['Gender'].fillna(train['Gender'].mode()[0])


train['Married'].fillna(train['Married'].mode()[0], inplace = True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace = True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace = True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace = True)


# In[266]:


train.isnull().sum()


# In[267]:


train['LoanAmount']=train['LoanAmount'].fillna(train['LoanAmount'].median())
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace = True)


# In[268]:


# To remove the skewness due to outliers, we use log transformation to get a normal distribution.
train['LoanAmount_log']= np.log(train['LoanAmount'])
test['LoanAmount_log']= np.log(test['LoanAmount'])
train['LoanAmount_log'].hist(bins = 20)


# In[269]:


train['Total_Income']= train['ApplicantIncome']+ train['CoapplicantIncome']
test['Total_Income']= test['ApplicantIncome']+ test['CoapplicantIncome']


# In[270]:


sns.displot(train['Total_Income'])


# In[271]:


train['Total_Income_log']=np.log(train['Total_Income'])
test['Total_Income_log']=np.log(test['Total_Income'])
sns.histplot(train['Total_Income_log'])


# In[272]:


train['EMI'] = train['LoanAmount']/train['Loan_Amount_Term']
test['EMI'] = test['LoanAmount']/test['Loan_Amount_Term']


# In[273]:


sns.displot(train['EMI'])


# In[274]:


# samp = train.drop(['Loan_ID','Gender','Married','Education','Property_Area','Self_Employed'], axis=1)
# plt.figure(figsize=(16,5))
# sns.heatmap(samp.corr(),annot=True)
# plt.title('Correlation Matrix (for Loan Status)')


# From the above figure, we can see that Credit_History (Independent Variable) has the maximum correlation with Loan_Status (Dependent Variable). Which denotes that the Loan_Status is heavily dependent on the Credit_History.
# 

# ## Final dataframe

# In[275]:


# train = train.drop(['LoanAmount_log','Total_Income','Total_Income_log','EMI'], axis=1)
# len(train.columns)
train.head()


# ### Comparison between Property Area for getting the Loan:

# In[276]:


# TBD


# 
# ### Prep: Let's replace the Variable values to Numerical form & display the Value Counts
# #### The data in Numerical form avoids disturbances in building the model.

# In[277]:


train['Loan_Status'].replace('Y',1,inplace=True)
train['Loan_Status'].replace('N',0,inplace=True)


# In[278]:


train['Loan_Status'].value_counts()


# In[279]:


train.Gender = train.Gender.map({'Male':1, 'Female':0})
train['Gender'].value_counts()


# In[280]:


train.Married=train.Married.map({'Yes':1,'No':0})
train['Married'].value_counts()


# In[281]:


train.Dependents=train.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
train['Dependents'].value_counts()


# In[282]:



train.Education=train.Education.map({'Graduate':1,'Not Graduate':0})
# train['Education'].value_counts()

train.Self_Employed=train.Self_Employed.map({'Yes':1,'No':0})
# train['Self_Employed'].value_counts()

train.Property_Area=train.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
train['Property_Area'].value_counts()

train['LoanAmount'].value_counts()

train['Loan_Amount_Term'].value_counts()
train.head()


# In[283]:


# train1 = train.drop(['Loan_ID'],axis=1) 
# plt.figure(figsize=(16,5))
# sns.heatmap(train1.corr(),annot=True)
# plt.title('Correlation Matrix (for Loan Status)')


# In[284]:


train = train.drop(['LoanAmount_log','Total_Income','Total_Income_log','EMI'], axis=1)


# In[285]:


train1 = train.drop(['Loan_ID'],axis=1) 
plt.figure(figsize=(16,5))
sns.heatmap(train1.corr(),annot=True)
plt.title('Correlation Matrix (for Loan Status)')


# # Model Training and Evaluation
# # As this is a classification problem so we will be using these models : 
# 
# KNeighborsClassifiers
# RandomForestClassifiers
# Support Vector Classifiers (SVC)
# Logistics Regression
# To predict the accuracy we will use the accuracy score function from scikit-learn library.

# In[288]:


# pip install scikit-learn


# ### Importing Packages for Classification algorithms
# 

# In[300]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


# In[298]:


train.head()


# In[303]:


X = train.iloc[1:,1:12].values
y = train.iloc[1:,12].values


# In[ ]:





# In[304]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)


# In[307]:


y_train


# ### Logistic Regression (LR)

# In[308]:


model = LogisticRegression()
model.fit(X_train,y_train)

lr_prediction = model.predict(X_test)
print('Logistic Regression accuracy = ', metrics.accuracy_score(lr_prediction,y_test))


# ### SVM

# In[312]:


model = svm.SVC()
model.fit(X_train,y_train)

svc_prediction = model.predict(X_test)
print('SVM accuracy = ', metrics.accuracy_score(svc_prediction,y_test))


# ### Decision Tree

# In[315]:


model = DecisionTreeClassifier()
model.fit(X_train,y_train)

dt_prediction = model.predict(X_test)
print('Decision Tree accuracy = ', metrics.accuracy_score(dt_prediction,y_test))


# ### K -NN

# In[316]:


model = KNeighborsClassifier()
model.fit(X_train,y_train)

knn_prediction = model.predict(X_test)
print('KNN accuracy = ', metrics.accuracy_score(knn_prediction,y_test))


# CONCLUSION:
# 
# The Logistic Regression algorithm gives us the maximum Accuracy (79% approx) compared to the other 3 Machine Learning Classification Algorithms

# In[ ]:




