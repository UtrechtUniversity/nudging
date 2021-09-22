"""Evaluate nudging model"""
from joblib import load

from nudging.train import read_data


def evaluate_probabilities(data):
    """Calculate accuracy of logistic regression model
    Args:
        data (pandas.DataFrame): dataframe with nudge success and probability
    Returns:
        int: accuracy in percentage
    """
    check = round(data['probability']) == data['success']
    correct = sum(check)
    total = check.shape[0]
    accuracy = int(round(correct*100/total, 0))
    print("Correct prediction of nudge success for {}% ({} out of {})". format(
        accuracy, correct, total))
    
    return accuracy


if __name__ == "__main__":

    # Read combined dataset, note this is the same as used for training
    features = ["nudge_domain", "age", "gender", "nudge_type"]
    df_nonan = read_data("data/interim/combined.csv", features)

    # Load model
    model = load("models/nudging.joblib")

    # Calculate probabilities and check results
    dataset = df_nonan.assign(probability=model.predict_proba(df_nonan[features])[:, 1])
    evaluate_probabilities(dataset)
