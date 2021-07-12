#!/usr/bin/env python
"""Use Naive Bayes classifier"""
import fileinput
import glob
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier


def combine():
    """Combine csv files"""
    filenames = glob.glob('data/interim/002*.csv')
    combined_file = 'data/processed/combined.csv'
    with open(combined_file, 'w') as fout, fileinput.input(filenames) as fin:
        for line in fin:
            fout.write(line)
    # Read in as DataFrame
    dataset = pd.read_csv(combined_file)
    # Drop rows with missing values
    dataset.replace("", float("NaN"), inplace=True)
    dataset.dropna(subset=["age", "gender"], inplace=True)

    return dataset


def classify(dataset):
    """Run NB classifier"""
    nudge_type = [np.array(json.loads(x)) for x in dataset['nudge_type']]
    target = [nudge_type[i] + 100*(1.0 - data) for i, data in enumerate(dataset['success'])]
    mlb = MultiLabelBinarizer()
    target = mlb.fit_transform(target)
    print("Classes: ", mlb.classes_)
    age = np.floor(dataset['age']/10.).astype(int)
    gender = np.array(dataset['gender']).astype(int)
    nudge_domain = np.array(dataset['nudge_domain']).astype(int)

    data = np.transpose(np.stack([age, gender, nudge_domain]))

    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.25)

    # Create a Gaussian Classifier
    clf = OneVsRestClassifier(CategoricalNB())

    # Train the model using the training sets
    clf.fit(data, target)

    # Predict the response for test dataset
    y_pred = clf.predict(data)

    nudge_domain = 5
    print("10-20 years, male: ", clf.predict([[1, 1, nudge_domain]]))
    print("20-30 years, male: ", clf.predict([[2, 1, nudge_domain]]))
    print("30-40 years, male: ", clf.predict([[3, 1, nudge_domain]]))
    print("40-50 years, male: ", clf.predict([[4, 1, nudge_domain]]))
    print("50-60 years, male: ", clf.predict([[5, 1, nudge_domain]]))

    print("10-20 years, female: ", clf.predict([[1, 0, nudge_domain]]))
    print("20-30 years, female: ", clf.predict([[2, 0, nudge_domain]]))
    print("30-40 years, female: ", clf.predict([[3, 0, nudge_domain]]))
    print("40-50 years, female: ", clf.predict([[4, 0, nudge_domain]]))
    print("50-60 years, female: ", clf.predict([[5, 0, nudge_domain]]))
    print("")

    # Model Accuracy, how often is the classifier correct?
    print("Accuracy CategoricalNB:", metrics.accuracy_score(target, y_pred))

    # test categorical
    clf = OneVsRestClassifier(GaussianNB())
    # Train the model using the training sets
    clf.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy GaussianNB:", metrics.accuracy_score(y_test, y_pred))

    # test multinomial
    clf = OneVsRestClassifier(MultinomialNB())
    # Train the model using the training sets
    clf.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy MultinomialNB:", metrics.accuracy_score(y_test, y_pred))

if __name__ == "__main__":

    combined_data = combine()
    classify(combined_data)
