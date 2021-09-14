#!/usr/bin/env python
"""Use Naive Bayes classifier"""
import os
import glob
import sys
from itertools import product

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
import matplotlib.pyplot as plt


def get_probability(data, predictors, method="logistic_regression"):
    """Calculate propability of nudge success with logistic regression
    Args:
        data (pandas.DataFrame): dataframe with nudge success per subject
        predictors (list): list with predictor variable names
        method (str): Choose logistic_regression or naive_bayes
    Returns:
        pandas.DataFrame: dataframe with probabilities
    Raises:
        RuntimeError: if requested method is unknown
    """
    if method == "naive_bayes":
        alg = CategoricalNB()
    elif method == "logistic_regression":
        alg = LogisticRegression()
    else:
        raise RuntimeError('Unknowm method, choose logistic_regression or naive_bayes')
    # Drop rows with missing values for predictors
    df_nonan = data.dropna(subset=predictors, inplace=False)
    model = alg.fit(
        df_nonan[predictors].to_numpy().astype('int'), df_nonan["success"].to_numpy().astype('int'))
    # Calculate probabilities for all subjects and check results
    data = df_nonan.assign(probability=model.predict_proba(df_nonan[predictors])[:, 1])
    check_prob(data)

    # Calculate probabilties for all subgroups and write to file
    subgroups = df_nonan[predictors].drop_duplicates().sort_values(by=predictors, ignore_index=True)
    result = subgroups.assign(
        probability=model.predict_proba(subgroups)[:, 1], success=model.predict(subgroups))
    result.to_csv("data/processed/nudge_probabilty.csv", index=False)

    return result


def check_prob(data):
    """Calculate accuracy of logistic regression model
    Args:
        data (pandas.DataFrame): dataframe with nudge success and probability
    """
    check = round(data['probability']) == data['success']
    correct = sum(check)
    total = check.shape[0]
    print("Correct prediction of nudge success for {}% ({} out of {})". format(
        int(round(correct*100/total, 0)), correct, total))


def plot_probability(data, labels):
    """Plot nudge effectiveness (probability of success) as function of age for each nudge type
    Args:
       data (pandas.DataFrame): dataframe with nudge effectiveness
       labels (dict): dict with labels to choose what to plot
    """
    width = 0.2
    _, axis = plt.subplots()
    colors = ["b", "r", "g", "c", "m", "y", "k"]
    types = data['nudge_type'].unique()
    position = 0
    condition = True
    for label in labels:
        condition = (data[label] == labels[label]) & condition

    for i, nudge in enumerate(types):
        dataset = data[(data['nudge_type'] == nudge) & condition]
        if dataset.empty:
            continue
        label = "nudge {}".format(nudge)
        dataset.plot(
            ax=axis, kind='bar', x='age', y='probability', label=label,
            color=colors[i], width=width, position=position)
        position += 1
    axis.set_xlabel('age [decades]')
    axis.set_ylabel('probability of nudge success')
    title = "{}".format(labels)
    plt.title(title)
    plt.show()


def flatten(data):
    """ Flatten list of tuples or tuple of tuples
    Args:
        data (list or tuple): list/tuple of data to flatten
    Returns:
        list: flattened list
    """
    result = []
    for var in data:
        if isinstance(var, tuple):
            result = flatten(var)
        else:
            result.append(var)

    return result


def plot_data(data, predictors):
    """Make plots of nudge effectiveness for each subject subgroup
    Args:
       predictors (list): list of predictors
       data (pandas.DataFrame): dataframe with nudge effectivenedd for all subgroups
    """
    plot_dict = {}
    for predictor in predictors:
        if predictor not in ["nudge_type", "age"]:
            plot_dict[predictor] = data[predictor].unique()

    for i, value in enumerate(plot_dict.values()):
        if i == 0:
            prod = value
        else:
            prod = product(prod, value)

    for term in prod:
        if isinstance(term, tuple):
            label = flatten(term)
        else:
            label = [term]
        labels = {key: label[i] for i, key in enumerate(plot_dict.keys())}
        plot_probability(data, labels)


if __name__ == "__main__":

    # Cleanup old data
    outdirs = ["data/processed"]
    # Make sure output dirs exist and are empty
    for dir_ in outdirs:
        if os.path.exists(dir_):
            files = glob.glob(f"{dir}/*")
            for f in files:
                os.remove(f)
        else:
            os.mkdir(dir_)

    combined_data = pd.read_csv("data/interim/combined.csv", encoding="iso-8859-1")
    features = ["nudge_domain", "age", "gender", "nudge_type"]
    # Use commandline specified algorithm if given or default
    if len(sys.argv) > 1:
        algorithm = sys.argv[1]
        data_frame = get_probability(combined_data, features, algorithm)
    else:
        data_frame = get_probability(combined_data, features)

    plot_data(data_frame, features)
