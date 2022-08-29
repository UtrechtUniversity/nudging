"""Evaluate nudging model"""
from joblib import load
import yaml

from nudging.train import read_data


def evaluate_probabilities(data):
    """Calculate accuracy of logistic regression model
    Args:
        data (pandas.DataFrame): dataframe with nudge success and probability
    Returns:
        int: accuracy in percentage
    """
    check = round(data['probability']) == data['outcome']
    correct = sum(check)
    total = check.shape[0]
    accuracy = int(round(correct*100/total, 0))
    print(f"Correct prediction of nudge success for {accuracy}% ({correct} out of {total})")

    return accuracy


if __name__ == "__main__":

    # Read combined dataset, note this is the same as used for training
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    features = config["features"]
    df_nonan = read_data("data/interim/combined.csv", features)

    # Load model
    model = load("models/nudging.joblib")
    model.predictors = config["features"]

    # Calculate probabilities and check results
    dataset = df_nonan.assign(probability=model.predict_outcome(df_nonan))
    evaluate_probabilities(dataset)
