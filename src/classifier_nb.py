#!/usr/bin/env python
"""Use Naive Bayes classifier"""
import fileinput
import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn import metrics


def combine():
    """Combine csv files"""
    filenames = glob.glob('data/processed/dataset_*.csv')
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
    target = np.array(dataset['success']).astype(int)
    target = np.array(
        dataset['nudge_type']).astype(int) + 100*(1.0 - np.array(dataset['success']).astype(int))
    print("nudge type==3: ", len(target[target == 3]))
    print("nudge type==103: ", len(target[target == 103]))
    print("nudge type==4: ", len(target[target == 4]))
    print("nudge type==104: ", len(target[target == 104]))
    print("")
    age = np.floor(dataset['age']/10.0).astype(int)
    gender = np.array(dataset['gender']).astype(int)
    nudge_domain = np.array(dataset['nudge_domain']).astype(int)

    data = np.transpose(np.stack([age, gender, nudge_domain]))

    # Split dataset into training set and test set
    x_train, x_test, y_train, y_test = train_test_split(
        data, target, test_size=0.25)

    # Create a Gaussian Classifier
    gnb = GaussianNB()

    # Train the model using the training sets
    gnb.fit(data, target)

    # Predict the response for test dataset
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
    print("Accuracy GaussianNB:", metrics.accuracy_score(target, y_pred))

    clf = CategoricalNB()
    # Train the model using the training sets
    clf.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy CategoricalNB:", metrics.accuracy_score(y_test, y_pred))


if __name__ == "__main__":

    combined_data = combine()
    classify(combined_data)
