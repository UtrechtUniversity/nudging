#!/usr/bin/env python
"""Use Naive Bayes classifier"""
import fileinput
import glob
import json

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def get_probability(df, predictors):
    """Calculate propability of nudge success with logistic regression
    Args:
        df (pandas.DataFrame):
        predictors (list): list with predictor variable names
    Returns:
        pandas.DataFrame: dataframe with probabilities
    """
    T = 'success'
    # Drop rows with missing values for predictors
    df_nonan = df.dropna(subset=predictors, inplace=False)
    lr_model = LogisticRegression().fit(df_nonan[predictors].to_numpy().astype('int'), df_nonan[T].to_numpy().astype('int'))
    data = df_nonan.assign(probability=lr_model.predict_proba(df_nonan[predictors])[:, 1])
    # write to file
    data.to_csv("data/processed/nudge_probabilty.csv", index=False)

    return data

def check_prob(df):
    """Calculate accuracy of logistic regression model
    Args:
        df (pandas.DataFrame): dataframe with nudge success and probability
    """
    check = round(df['probability'])==df['success']
    correct = sum(check)
    total = check.shape[0]
    print("Correct prediction of nudge success for {}% ({} out of {})". format(int(round(correct*100/total, 0)), correct, total))

def combine():
    """Combine csv files"""
    filenames = glob.glob('data/interim/*.csv')
    dataframes = []
    for fin in filenames:
        print(fin)
        dataframes.append(pd.read_csv(fin, encoding="iso-8859-1"))
    dataset = pd.concat(dataframes)

    # Replace missing values with Nans
    dataset.replace("", pd.NA, inplace=True)

    # Convert age to decades
    dataset.age = (dataset.age/10).astype(int)
    dataset.to_csv("data/processed/combined.csv", index=False)
    return dataset


if __name__ == "__main__":

    combined_data = combine()
    predictors = ['age', 'gender', 'nudge_type', 'nudge_domain']
    result = get_probability(combined_data, predictors)

    check_prob(result)
    print(result)
