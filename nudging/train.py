"""Train nudging model using probabilistic classifier"""
import glob

import category_encoders as ce
from joblib import dump
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression, SGDRegressor
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import yaml

from nudging.model import BiRegressor, MonoRegressor, XRegressor
from nudging.model.probmodel import ProbModel
from nudging.utils import clean_dirs, read_data

regressors = {
    "gauss_process": GaussianProcessRegressor,
    "ridge": Ridge,
    "linear": LinearRegression,
    "logistic": LogisticRegression,
    "nb": GaussianNB,
    "sgd": SGDRegressor,
    "elasticnet": ElasticNet,
    "ard": ARDRegression,
    "bayesian_ridge": BayesianRidge,
    "knn": KNeighborsRegressor,
    "mlp": MLPRegressor,
    "svm": SVR,
    "decision_tree": DecisionTreeRegressor,
    "extra_tree": ExtraTreeRegressor,
}

learner_dict = {
    "s-learner": MonoRegressor,
    "t-learner": BiRegressor,
    "x-learner": XRegressor,
    "probabilistic": ProbModel
}

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
    model = learner_dict[config["learner_type"]](regressor())
    # model = BiRegressor(BayesianRidge())
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
