"""Train nudging model using probabilistic classifier"""
import glob

import pandas as pd
# from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from joblib import dump
import yaml

from nudging.model.biregressor import BiRegressor
# from nudging.model.probmodel import ProbModel
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
    model = BiRegressor(BayesianRidge())
    # model = ProbModel(LogisticRegression())
    model.predictors = config["features"]

    # combine separate datasets to one
    files = glob.glob('data/interim/[!combined.csv]*')
    datasets = []
    for file_path in files:
        data = read_data(file_path, predictors)
        if data.empty is False:
            datasets.append(model.preprocess(data))

    dataset = pd.concat(datasets)
    # Use age in decennia
    dataset["age"] = (dataset["age"]/10.0).astype({'age': 'int32'})

    # train model
    model.train(dataset)

    # Save trained model
    dump(model, "models/nudging.joblib")
    print("Model saved to models/nudging.joblib")

    # save training data
    dataset.to_csv('data/interim/combined.csv')
