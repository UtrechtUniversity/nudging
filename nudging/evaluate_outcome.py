"""Evaluate outcome (+CATE) of datasets"""

from scipy.stats import spearmanr
import numpy as np


def safe_spearmanr(arr_a, arr_b):
    "Compute the spearman-R correlation, but 0 if all equal"
    if np.all(arr_a[0] == arr_a) or np.all(arr_b[0] == arr_b):
        return 0
    return spearmanr(arr_a, arr_b).correlation


def evaluate_outcome(model, dataset, k=5, n=1):
    """Evaluate the outcome of a model with a dataset

    Arguments
    ---------
    model: BaseModel
        model to be trained and evaluated on the dataset.
    dataset: BaseDataset
        Dataset on which the model is evaluated.
    k: int
        Number of folds
    n: int
        Number of iterations to evaluate over
    """
    results = []
    for _ in range(n):
        for train_data, test_data in dataset.kfolds(k=k):
            model.train(model.preprocess(train_data.standard_df))
            model_outcome = model.predict_outcome(test_data.standard_df)
            if (np.all(model_outcome == model_outcome[0])
                    or np.all(test_data.outcome == test_data.outcome[0])):
                corr = 0
            else:
                corr = spearmanr(model_outcome, test_data.outcome).correlation
            results.append(corr)
    return results


def evaluate_performance(model, dataset, k=5, n=1):
    """Evaluate the outcome + CATE of a model with a dataset

    Arguments
    ---------
    model: BaseModel
        model to be trained and evaluated on the dataset.
    dataset: BaseDataset
        Dataset on which the model is evaluated.
    k: int
        Number of folds
    n: int
        Number of iterations to evaluate over
    """

    cate_corr = []
    outcome_corr = []
    for _ in range(n):
        for train_data, test_data in dataset.kfolds(k=k):
            model.train(model.preprocess(train_data.standard_df))
            test_df = test_data.standard_df
            cate = model.predict_cate(test_df)
            outcome = model.predict_outcome(test_df)
            cate_corr.append(safe_spearmanr(cate, test_data.cate))
            outcome_corr.append(safe_spearmanr(outcome, test_data.outcome))
    return cate_corr, outcome_corr
