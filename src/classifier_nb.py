#!/usr/bin/env python
"""Use Naive Bayes classifier"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics

df = pd.read_csv("test.csv")
target = np.array(df['nudge_success']).astype(int)
target = np.array(
    df['nudge_type']).astype(int) + 100*(1.0 - np.array(df['nudge_success']).astype(int))
print("nudge type==3: ", len(target[target==3]))
print("nudge type==103: ", len(target[target==103]))
print("nudge type==4: ", len(target[target==4]))
print("nudge type==104: ", len(target[target==104]))
print("")
age = np.floor(df['age']/10.0).astype(int)
gender = np.array(df['gender']).astype(int)
nudge_domain = np.array(df['nudge_domain']).astype(int)

data = np.transpose(np.stack([age, gender, nudge_domain]))


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.25)#, random_state=1) # 70% training and 30% test

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(data, target)

#Predict the response for test dataset
y_pred = gnb.predict(data)

print("10-20 years, male: ", gnb.predict([[1, 1, 3]]))
print("20-30 years, male: ", gnb.predict([[2, 1, 3]]))
print("30-40 years, male: ", gnb.predict([[3, 1, 3]]))
print("40-50 years, male: ", gnb.predict([[4, 1, 3]]))
print("50-60 years, male: ", gnb.predict([[5, 1, 3]]))

print("10-20 years, female: ", gnb.predict([[1, 0, 3]]))
print("20-30 years, female: ", gnb.predict([[2, 0, 3]]))
print("30-40 years, female: ", gnb.predict([[3, 0, 3]]))
print("40-50 years, female: ", gnb.predict([[4, 0, 3]]))
print("50-60 years, female: ", gnb.predict([[5, 0, 3]]))
print("")

# Model Accuracy, how often is the classifier correct?
print("Accuracy GaussianNB:",metrics.accuracy_score(target, y_pred))

clf = CategoricalNB()
#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy CategoricalNB:",metrics.accuracy_score(y_test, y_pred))
