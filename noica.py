# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:15:54 2020

@author: Al Rahrooh
"""

#dimension reduction is 128 x 5 for Self Paced
import pandas as pd
from random import sample
import numpy as np

#dimension reduction is 128 x 5 for Speed 50

df_S1_50 = pd.read_csv('S1_50.csv')
df_S1_50 = df_S1_50.transpose()
df_S1_50_1 = df_S1_50[50000:100000]

subset_50 = df_S1_50_1.sample(n=5)
subset_Speed50_S1 = subset_50.transpose() 


#dimension reduction is 128 x 5 for Speed 75

df_S1_75 = pd.read_csv('S1_75.csv')
df_S1_75 = df_S1_75.transpose()
df_S1_75_1 = df_S1_75[50000:100000]


subset_75 = df_S1_75_1.sample(n=5)
subset_Speed75_S1 = subset_75.transpose() 


#dimension reduction is 128 x 5 for Speed 100

df_S1_100 = pd.read_csv('S1_100.csv')
df_S1_100 = df_S1_100.transpose()
df_S1_100_1 = df_S1_100[50000:100000]

subset_100 = df_S1_100_1.sample(n=5)
subset_Speed100_S1 = subset_100.transpose() 


#dimension reduction is 128 x 5 for Speed 125
df_S1_125 = pd.read_csv('S1_125.csv')
df_S1_125 = df_S1_125.transpose()
df_S1_125_1 = df_S1_125[50000:100000]

subset_125 = df_S1_125_1.sample(n=5)
subset_Speed125_S1 = subset_125.transpose() 


#self paced
df_S1_SP = pd.read_csv('S1_SP.csv')
df_S1_SP = df_S1_SP.transpose()
df_S1_SP_1 = df_S1_SP[50000:100000]

subset_SP = df_S1_SP_1.sample(n=5)
subset_SP_S1 = subset_SP.transpose()   


###transforming
X1_transformed = pd.DataFrame(subset_Speed50_S1)
X2_transformed = pd.DataFrame(subset_Speed75_S1)
X3_transformed = pd.DataFrame(subset_Speed100_S1)
X4_transformed = pd.DataFrame(subset_Speed125_S1)
X5_transformed = pd.DataFrame(subset_SP_S1)

X = pd.concat([X1_transformed, X2_transformed, X3_transformed, X4_transformed, X5_transformed])

df_speed1 = pd.DataFrame(["Speed1"])
df_speed1 = pd.concat([df_speed1]*127)

df_speed2 = pd.DataFrame(["Speed2"])
df_speed2 = pd.concat([df_speed2]*127)

df_speed3 = pd.DataFrame(["Speed3"])
df_speed3 = pd.concat([df_speed3]*127)

df_speed4 = pd.DataFrame(["Speed4"])
df_speed4 = pd.concat([df_speed4]*127)

df_speed5 = pd.DataFrame(["Speed5"])
df_speed5 = pd.concat([df_speed5]*127)

y = pd.concat([df_speed1,df_speed2,df_speed3,df_speed4,df_speed5])

y = np.array(y)
X = np.array(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

seed = 128
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 50
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)

y_pred = model.fit(X_train, y_train).predict(X_test)
confusion_matrix(y_test, y_pred)



# Random Forest Classification
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

seed = 128
num_trees = 50
max_features = 5
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)

y_pred = model.fit(X_train, y_train).predict(X_test)
confusion_matrix(y_test, y_pred)


# AdaBoost Classification
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix

seed = 128
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())

y_pred = model.fit(X_train,y_train).predict(X_test)
confusion_matrix(y_test,y_pred)


# Logistic Regression Classification
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn import metrics

lr = linear_model.LogisticRegression()
y_pred = lr.fit(X_train, y_train)
y_pred = model.fit(X_train,y_train).predict(X_test)
confusion_matrix(y_test,y_pred)


# SVM Classification linear
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


classifier = svm.SVC(kernel='linear', C=0.50)
y_pred = classifier.fit(X_train, y_train).predict(X_test)
confusion_matrix(y_test, y_pred)

# SVM Classification rbf
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


classifier = svm.SVC(kernel='rbf', C=0.50)
y_pred = classifier.fit(X_train, y_train).predict(X_test)
confusion_matrix(y_test, y_pred)

# Gaussian Classification
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

confusion_matrix(y_test,y_pred)




