#!/usr/bin/env python
"""Use Naive Bayes classifier"""
import fileinput
import json
import sys
from itertools import product

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
import matplotlib.pyplot as plt


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
        raise RuntimeError('Unknowm algorithm, choose logistic_regresiion or naive_bayes')
    T = "success"
    # Drop rows with missing values for predictors
    df_nonan = df.dropna(subset=predictors, inplace=False)
    model = alg.fit(df_nonan[predictors].to_numpy().astype('int'), df_nonan[T].to_numpy().astype('int'))
    # Calculate probabilties for all subjects and check results
    data = df_nonan.assign(probability=model.predict_proba(df_nonan[predictors])[:, 1])
    check_prob(data)
    
    # Calculate probabilties for all subgroups and write to file
    subgroups = df_nonan[predictors].drop_duplicates().sort_values(by=predictors, ignore_index=True)
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


def plot_probability(labels):
    """Plot propability of nudge success as function of age
    Args:
       labels (dict): dict with labels for plotting
    """    
    width = 0.2
    fig, ax = plt.subplots()
    colors = ["b", "r", "g", "c", "m", "y", "k"]
    types = result['nudge_type'].unique()
    position = 0
    condition = True
    for label in labels:
        condition = (result[label]==labels[label]) & condition

    for i, nudge in enumerate(types): 
        df = result[(result['nudge_type']==nudge) & condition]
        if df.empty:
            continue
        label = "nudge {}".format(nudge)
        df.plot(ax=ax, kind='bar', x='age', y='probability', label=label, color=colors[i], width=width, position=position)
        position += 1
    ax.set_xlabel('age [decades]')
    ax.set_ylabel('probability of nudge success')
    title = "{}".format(labels)
    plt.title(title)
    plt.show()


def flatten(data):
    result = []
    for x in data:
        if isinstance(x, tuple):
            result = flatten(x)
        else:    
            result.append(x)

    return result


if __name__ == "__main__":

    combined_data = pd.read_csv("data/processed/combined.csv", encoding="iso-8859-1")
    predictors = ["nudge_domain", "age", "gender", "nudge_type"]
    algorithm = "logistic_regression" # choose naive_bayes or logistic regression"
    if len(sys.argv) > 1:
        algorithm = sys.argv[1]
    result = get_probability(combined_data, predictors, algorithm)

    # Plot results
    plot_dict = {}
    for predictor in predictors:
        if predictor not in ["nudge_type", "age"]:
            plot_dict[predictor] = result[predictor].unique()
 
    for i, value in enumerate(plot_dict.values()):
        if i == 0:
            prod = value
        else:
            prod = product(prod, value)

    for p in prod:
        if isinstance(p, tuple):
            label = flatten(p)
        else:
            label = [p]
        labels = {key: label[i] for i, key in enumerate(plot_dict.keys())}
        plot_probability(labels)

    # Todo: Classify: get per subject subgroup, the most effective nudge