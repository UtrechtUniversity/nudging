"""Train nudging model using probabilistic classifier"""
import glob

import category_encoders as ce
from joblib import dump
import pandas as pd
import yaml

from nudging.model import regressors, learners
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

    regressor = regressors[config["model_name"]]
    model = learners[config["learner_type"]](regressor())

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

    # Encode categories
    encode = config.get("encode", None)
    if encode:
        encoder = ce.OneHotEncoder(cols=encode)
        dataset = encoder.fit_transform(dataset)
        predictors = list(dataset.columns)
        predictors.remove("outcome")
        if "nudge" in predictors:
            predictors.remove("nudge")

    # train model
    model.predictors = predictors
    model.train(dataset)

    # Save trained model
    dump(model, "models/nudging.joblib")
    print("Model saved to models/nudging.joblib")

    # save training data
    dataset.to_csv('data/interim/combined.csv')
