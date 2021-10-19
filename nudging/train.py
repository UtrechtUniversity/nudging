"""Train nudging model using probabilistic classifier"""
import sys
import glob

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from joblib import dump
import yaml

from nudging.model.probmodel import ProbModel
from nudging.utils import clean_dirs, read_data


def train_model(data, predictors_, model):
    """Calculate propability of nudge success with logistic regression
    Args:
        data (pandas.DataFrame): dataframe with nudge success per subject
        predictors_ (list): list with predictor variable names
        model (BaseModel): ML model to use for training
    Returns:
        model: trained model
    Raises:
        RuntimeError: if requested method is unknown
    """
    # Drop rows with missing values for predictors
    df_nonan = data.dropna(subset=predictors_, inplace=False)

    model.fit(
        df_nonan[predictors_].to_numpy().astype('int'),
        df_nonan["success"].to_numpy().astype('int'))

    return model


if __name__ == "__main__":

    # Cleanup old data
    outdirs = ["models"]
    # Make sure output dirs exist and are empty
    clean_dirs(outdirs)

    # Get predictors from config.yaml
    config = yaml.safe_load(open("config.yaml"))
    predictors = config["features"]

    # Choose model
    # model = BaseBiRegressor(BayesianRidge)
    model = ProbModel(LogisticRegression)

    # combine separate datasets to one
    files = glob.glob('data/interim/[!combined.csv]*')
    datasets = []
    for file_path in files:
        data = read_data(file_path, predictors)
        datasets.append(model.preprocess(data))

    dataset = pd.concat(datasets)

    # train model
    nudging_model = train_model(dataset, predictors, model)

    # Save trained model
    dump(nudging_model, "models/nudging.joblib")
    print("Model saved to models/nudging.joblib")

    # save training data
    dataset.to_csv('data/interim/combined.csv')