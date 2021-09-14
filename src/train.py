"""Train nudging model using probabilistic classifier"""
import os
import glob
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from joblib import dump


def read_data(filename, features):
    """ Read data from file and return dataframe for selected features
    Args:
        filename (str): name of csv file
        features (list): list of string with feature names
    """
    # Read combined dataset, note this is the same as used for training
    data = pd.read_csv(filename, encoding="iso-8859-1")
    # Drop rows with missing values for features
    df_nonan = data.dropna(subset=features, inplace=False)

    return df_nonan


def train_model(data, predictors_, method="logistic_regression"):
    """Calculate propability of nudge success with logistic regression
    Args:
        data (pandas.DataFrame): dataframe with nudge success per subject
        predictors_ (list): list with predictor variable names
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
    df_nonan = data.dropna(subset=predictors_, inplace=False)
    model = alg.fit(
        df_nonan[predictors_].to_numpy().astype('int'),
        df_nonan["success"].to_numpy().astype('int'))

    return model


if __name__ == "__main__":

    # Cleanup old data
    outdirs = ["models"]
    # Make sure output dirs exist and are empty
    for dir_ in outdirs:
        if os.path.exists(dir_):
            files = glob.glob(f"{dir}/*")
            for f in files:
                os.remove(f)
        else:
            os.mkdir(dir_)

    predictors = ["nudge_domain", "age", "gender", "nudge_type"]
    dataset = read_data("data/interim/combined.csv", predictors)

    # Use commandline specified algorithm if given or default
    if len(sys.argv) > 1:
        algorithm = sys.argv[1]
        nudging_model = train_model(dataset, predictors, algorithm)
    else:
        nudging_model = train_model(dataset, predictors)

    # Save trained model
    dump(nudging_model, "models/nudging.joblib")
