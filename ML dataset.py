#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sys
print('Pyhton: {}'.format(sys.version))
import scipy
print('Scipy: {}'.format(scipy.__version__))
import numpy
print('Numpy: {} '.format(numpy.__version__))
import matplotlib
print('Matplotlib: {}'.format(matplotlib.__version__))
import pandas
print('Pandas: {}'.format(pandas.__version__))
import sklearn
print('Sklearn: {}'.format(sklearn.__version__))


# In[9]:


import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[14]:


names=['sepel-length','sepel-width','petal-length','petal-width','class']
dataset=pandas.read_csv('iris.data',names=names)
#No of rows & cols
print(dataset.shape)


# In[15]:


print(dataset.head(30))# 30first data


# In[16]:


#Summary of each attribute
print(dataset.describe())


# In[17]:


#No of instances belonging to each class
print(dataset.groupby('class').size())


# In[20]:


# some visulization
# we will create 2 diff types of plot-univariate(understand each attribute) and
#multi-variate(to understand reln b/w each attribute) plot

#1.univa=Box and viscous plot to get idea of i/p
dataset.plot(kind='box', subplots=True , layout=(2,2), sharex=False , sharey=False)
#sharex/y=false as dont want to share it along any x and y to any visualization
plt.show()


# In[21]:


#Creating histogram to get more clear idea of i/p
dataset.hist()
plt.show()


# In[22]:


#Multivariate plot-to seee the interaction b/w the diff variable
#for creating Scatter plot we need scatter matrix
scatter_matrix(dataset)
plt.show()


# In[25]:


#Let's create our dataset and estimate accuracy based on unseen data
#we will be creating some model and estimating the accuracy
#Validation dataset =Training dataset ,will be used to train our data set
#We will split our data - 1.Training(80% will be used here) 2.Unseendata
array=dataset.values
X=array[:,0:4]
Y=array[:,4]
validation_size=0.20
seed=6 #This method seed sets the integer starting value used in generating random values
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)
                            #Selecting some model and splting value of X and Y


# In[26]:


#Now we create test harness:split data 10 parts,train on 9 part and test in 1 part
seed=6
scoring='accuracy'#Using metric of accuracy to evaluate the model-
#It is ratio of correctly predicted instances/total instances in the dataset X


# In[31]:


#Building the model-5 diff models 
#Evaluate model in each turn
models=[]
models.append(('LR',LogisticRegression()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))
#evaluate each model in turn-to check the best accuracy of which model
results=[]
names=[]
for name,model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=seed)
    cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg="%s: %f (%f)"%(name,cv_results.mean(),cv_results.std())
    print(msg)
    


# In[ ]:


#Since LDA gives best accuracy-it will be best for PREDICTION

