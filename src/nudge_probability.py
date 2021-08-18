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
from sklearn.linear_model import LogisticRegression


def get_probability(df):
    """Calculate propensity score with logistic regression
    Args:
        df (pandas.DataFrame):
    
    Returns:
        pandas.DataFrame: dataframe with propensity score
    """
    T = 'success'
    Y = ['education','hhi','ethnicity','political_party','SciKnow','MMS','CRT_ACC']
    X = df.columns.drop([T]+Y)  
    lg_model = LogisticRegression().fit(df[X].to_numpy().astype('int'), df[T].to_numpy().astype('int'))

    data = df.assign(probability=lg_model.predict_proba(df[X])[:, 1])
    # write to file
    data.to_csv("data/processed/nudge_probabilty.csv", index=False)

    return data

def combine():
    """Combine csv files"""
    filenames = glob.glob('data/interim/*.csv')
    dataframes = []
    for fin in filenames:
        print(fin)
        dataframes.append(pd.read_csv(fin, encoding="iso-8859-1"))
    dataset = pd.concat(dataframes)

    # Drop rows with missing values
    dataset.replace("", float("NaN"), inplace=True)
    dataset.dropna(subset=["age", "gender"], inplace=True)
    dataset.to_csv("data/processed/combined.csv", index=False)
    return dataset


if __name__ == "__main__":

    combined_data = combine()
    result = get_probability(combined_data)
    print(result)
