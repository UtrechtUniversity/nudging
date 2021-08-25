#!/usr/bin/env python
"""Use Naive Bayes classifier"""
import fileinput
import glob
import json

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB


def get_probability(df, predictors, algorithm):
    """Calculate propability of nudge success with logistic regression
    Args:
        df (pandas.DataFrame):
        predictors (list): list with predictor variable names
        algorithm (str): Choose naive_bayes or logistic_regression
    Returns:
        pandas.DataFrame: dataframe with probabilities
    Raises:
        RuntimeError: if requested algorithm is unknown
    """
    if algorithm == "naive_bayes":
        alg = CategoricalNB()
    elif algorithm == "logistic_regression":
        alg = LogisticRegression()
    else:
        raise RuntimeError('Unknowm algorithm')
    T = "success"
    # Drop rows with missing values for predictors
    df_nonan = df.dropna(subset=predictors, inplace=False)
    model = alg.fit(df_nonan[predictors].to_numpy().astype('int'), df_nonan[T].to_numpy().astype('int'))
    # Calculate probabilties for all subjects and check results
    data = df_nonan.assign(probability=model.predict_proba(df_nonan[predictors])[:, 1])
    check_prob(data)
    
    # Calculate probabilties for all subgroups and write to file
    subgroups = df[predictors].drop_duplicates().sort_values(by=predictors, ignore_index=True)
    result = subgroups.assign(probability=model.predict_proba(subgroups)[:, 1], success=model.predict(subgroups))
    result.to_csv("data/processed/nudge_probabilty.csv", index=False)

    return result


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
    predictors = ["nudge_domain", "age", "gender", "nudge_type"]
    algorithm = "logistic_regression" # choose naive_bayes or logistic regression"
    result = get_probability(combined_data, predictors, algorithm)
    print(result)

    # Todo: Plot results

    # Todo: Classify: get per subject subgroup, the most effective nudge