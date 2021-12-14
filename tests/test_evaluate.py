import pandas as pd

from nudging.evaluate import evaluate_probabilities


def test_evaluate_probabilities():

    df = pd.DataFrame({
        'probability': [0.6, 0.9, 0.1, 0.8, 0.4, 0.8, 0.4, 0.6, 0.3, 0.4],
        'outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    }).reset_index()

    result = evaluate_probabilities(df)
    expected = 80

    assert result == expected
