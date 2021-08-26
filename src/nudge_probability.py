#!/usr/bin/env python
"""Use Naive Bayes classifier"""
import fileinput
import glob
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


def plot_probability(nudge_domain, gender):
    """Plot propability of nudge success as function of age
    Args:
       nudge_domain (int): category of nudge domain
    """    
    width = 0.2
    fig, ax = plt.subplots()
    colors = ["b", "r", "g", "c", "m", "y", "k"]
    types = result['nudge_type'].unique()
    position = 0
    for i, nudge in enumerate(types): 
        df = result[(result['nudge_type']==nudge) & (result['gender']==gender) & (result['nudge_domain']==nudge_domain)]
        if df.empty:
            continue
        label = "nudge {}".format(nudge)
        df.plot(ax=ax, kind='bar', x='age', y='probability', label=label, color=colors[i], width=width, position=position)
        position += 1
    ax.set_xlabel('age [decades]')
    ax.set_ylabel('probability of nudge success')
    title = "nudge domain {}, gender {}".format(nudge_domain, gender)
    plt.title(title)
    plt.show()


if __name__ == "__main__":

    combined_data = combine()
    predictors = ["nudge_domain", "age", "gender", "nudge_type"]
    algorithm = "logistic_regression" # choose naive_bayes or logistic regression"
    if len(sys.argv) > 1:
        algorithm = sys.argv[1]
    result = get_probability(combined_data, predictors, algorithm)

    # Plot results
    domains = result['nudge_domain'].unique()
    genders = result['gender'].unique()
    for nudge_domain, gender in list(product(domains, genders)):
        plot_probability(nudge_domain, gender)

    # Todo: Classify: get per subject subgroup, the most effective nudge