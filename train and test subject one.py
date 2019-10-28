# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:51:14 2019

@author: Al Rahrooh
"""

from sklearn.model_selection import train_test_split

#combine row wise 
import numpy as np
import pandas as pd

X1_transformed = pd.DataFrame(X1_transformed)
X2_transformed = pd.DataFrame(X2_transformed)
X3_transformed = pd.DataFrame(X3_transformed)
X4_transformed = pd.DataFrame(X4_transformed)
X5_transformed = pd.DataFrame(X5_transformed)

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
#split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# Bagged Decision Trees for Classification
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

seed = 128
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)

y_pred = model.fit(X_train, y_train).predict(X_test)
confusion_matrix(y_test, y_pred)



# Random Forest Classification
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

seed = 128
num_trees = 100
max_features = 5
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)

y_pred = model.fit(X_train, y_train).predict(X_test)
confusion_matrix(y_test, y_pred)



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


from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn import metrics

lr = linear_model.LogisticRegression()
y_pred = lr.fit(X_train, y_train)
y_pred = model.fit(X_train,y_train).predict(X_test)
confusion_matrix(y_test,y_pred)


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


classifier = svm.SVC(kernel='linear', C=0.50)
y_pred = classifier.fit(X_train, y_train).predict(X_test)
confusion_matrix(y_test, y_pred)

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

confusion_matrix(y_test,y_pred)


