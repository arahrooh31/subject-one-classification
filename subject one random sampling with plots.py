import pandas as pd
import numpy as np

#reading in data

df_S1_50 = pd.read_csv('S1_50.csv')
df_S1_75 = pd.read_csv('S1_75.csv')
df_S1_100 = pd.read_csv('S1_100.csv')
df_S1_125 = pd.read_csv('S1_125.csv')
df_S1_SP = pd.read_csv('S1_SP.csv')

df_S1_50 = df_S1_50.T
df_S1_50_1 = df_S1_50.sample(n=5)
df_S1_50_1 = df_S1_50_1.T


import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

fig = plt.figure()
df_S1_50_1.plot(legend = False)
plt.title('0.50 m/s')
plt.xlabel('EEG Channels')
plt.ylabel('EEG Voltage')

df_S1_50_1 = np.array(df_S1_50_1)



df_S1_75 = df_S1_75.T
df_S1_75_1 = df_S1_75.sample(n=5)
df_S1_75_1 = df_S1_75_1.T

fig = plt.figure()
df_S1_75_1.plot(legend = False)
plt.title('0.75 m/s')
plt.xlabel('EEG Channels')
plt.ylabel('EEG Voltage')

df_S1_75_1 = np.array(df_S1_75_1)



df_S1_100 = df_S1_100.T
df_S1_100_1 = df_S1_100.sample(n=5)
df_S1_100_1 = df_S1_100_1.T

fig = plt.figure()
df_S1_100_1.plot(legend = False)
plt.title('1.0 m/s')
plt.xlabel('EEG Channels')
plt.ylabel('EEG Voltage')

df_S1_100_1 = np.array(df_S1_100_1)



df_S1_125 = df_S1_125.T
df_S1_125_1 = df_S1_125.sample(n=5)
df_S1_125_1 = df_S1_125_1.T

fig = plt.figure()
df_S1_125_1.plot(legend = False)
plt.title('1.25 m/s')
plt.xlabel('EEG Channels')
plt.ylabel('EEG Voltage')

df_S1_125_1 = np.array(df_S1_125_1)

df_S1_SP = df_S1_SP.T
df_S1_SP_1 = df_S1_SP.sample(n=5)
df_S1_SP_1 = df_S1_SP_1.T


fig = plt.figure()
df_S1_SP_1.plot(legend = False)
plt.title('Self-Paced')
plt.xlabel('EEG Channels')
plt.ylabel('EEG Voltage')

df_S1_SP_1 = np.array(df_S1_SP_1)

#taking sample of 25 from rows
#we can vary the random sample 


X = np.vstack((df_S1_50_1,df_S1_75_1))
X = np.vstack((X, df_S1_100_1))
X = np.vstack((X, df_S1_125_1))
X = np.vstack((X, df_S1_SP_1))

#dimension is now 640 x 25

#this section is good to go
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

#we need to create this manually
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

seed = 54
kfold = model_selection.KFold(n_splits=50, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 10
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)

y_pred = model.fit(X_train, y_train).predict(X_test)
confusion_matrix(y_pred,y_test)



# Random Forest Classification
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

seed = 457
num_trees = 50
max_features = 5
kfold = model_selection.KFold(n_splits=5, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)

y_pred = model.fit(X_train, y_train).predict(X_test)
confusion_matrix(y_pred,y_test)


# AdaBoost Classification
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix

seed = 784
num_trees = 4
kfold = model_selection.KFold(n_splits=3, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
print(results.mean())

y_pred = model.fit(X_train,y_train).predict(X_test)
confusion_matrix(y_pred,y_test)


# Logistic Regression Classification
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn import metrics

lr = linear_model.LogisticRegression()
y_pred = lr.fit(X_train, y_train)
y_pred = model.fit(X_train,y_train).predict(X_test)
confusion_matrix(y_pred,y_test)


# SVM Classification
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


classifier = svm.SVC(kernel='linear', C=1)
y_pred = classifier.fit(X_train, y_train).predict(X_test)
confusion_matrix(y_pred, y_test)


# Gaussian Classification
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

confusion_matrix(y_pred,y_test)



# SVM RBF Classification
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


classifier = svm.SVC(kernel='rbf', C=1)
y_pred = classifier.fit(X_train, y_train).predict(X_test)
confusion_matrix(y_pred, y_test)

