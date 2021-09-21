"""Train nudging model using probabilistic classifier"""
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from joblib import dump

from utils import clean_dirs, read_data


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
    clean_dirs(outdirs)

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
    print("Model saved to models/nudging.joblib")
