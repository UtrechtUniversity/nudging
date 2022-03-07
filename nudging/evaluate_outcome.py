from scipy.stats import spearmanr
import numpy as np


def safe_spearmanr(a, b):
    if np.all(a[0] == a) or np.all(b[0] == b):
        return 0

    return spearmanr(a, b).correlation


def evaluate_outcome(model, dataset, k=5, n=1):
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
