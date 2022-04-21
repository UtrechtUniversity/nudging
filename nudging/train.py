"""Train nudging model using probabilistic classifier"""
import glob

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
# from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from joblib import dump
import yaml

from nudging.model.biregressor import BiRegressor
# from nudging.model.probmodel import ProbModel
from nudging.utils import clean_dirs, read_data


def encode_dataframe(data_frame, column):
    """Tranform dataframe by onehot encoding one of its columns"""
    jobs_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    transformed = jobs_encoder.fit_transform(data_frame[[column]])

    #Create a Pandas DataFrame of the hot encoded column
    ohe_df = pd.DataFrame(transformed, columns=jobs_encoder.get_feature_names())

    # One-hot encoding removed index; put it back
    ohe_df.index = data_frame.index

    #concat with original data
    result = pd.concat([data_frame, ohe_df], axis=1).drop([column], axis=1)

    return result

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
        dataset = encode_dataframe(dataset, encode)
        predictors = list(dataset.columns)
        predictors.remove("outcome")
        predictors.remove("nudge")

    # train model
    model.predictors = predictors
    model.train(dataset)

    # Save trained model
    dump(model, "models/nudging.joblib")
    print("Model saved to models/nudging.joblib")

    # save training data
    dataset.to_csv('data/interim/combined.csv')
