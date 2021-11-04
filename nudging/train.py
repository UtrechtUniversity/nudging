"""Train nudging model using probabilistic classifier"""
import glob

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB
from joblib import dump
import yaml

# from nudging.model.base import BaseBiRegressor
from nudging.model.probmodel import ProbModel
from nudging.utils import clean_dirs, read_data


if __name__ == "__main__":

    # Cleanup old data
    outdirs = ["models"]
    # Make sure output dirs exist and are empty
    clean_dirs(outdirs)

    # Get predictors from config.yaml
    config = yaml.safe_load(open("config.yaml"))
    predictors = config["features"]

    # Choose model
    # model = BaseBiRegressor(BayesianRidge())
    model = ProbModel(LogisticRegression())
    model.predictors = config["features"]

    # combine separate datasets to one
    files = glob.glob('data/interim/[!combined.csv]*')
    datasets = []
    for file_path in files:
        data = read_data(file_path, predictors)
        datasets.append(model.preprocess(data))

    dataset = pd.concat(datasets)

    # train model
    model.train(dataset)

    # Save trained model
    dump(model, "models/nudging.joblib")
    print("Model saved to models/nudging.joblib")

    # save training data
    dataset.to_csv('data/interim/combined.csv')
