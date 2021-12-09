"""Module for calculating conditional average treatment effect (cate) """
import pandas as pd
from scipy.stats import spearmanr
import numpy as np


def get_cate(model, dataset, k=10):
    """Compute the CATE for k-fold validation

    Arguments
    ---------
    dataset: BaseDataSet
        Dataset to train and validate on.
    model: BaseModel
        Model to train and validate with.
    k: int
        Number of folds

    Returns
    -------
    results: list[tuple]
        A set of results for each of the folds, where one result consists of
        a list of cate estimates and a list of indices to which the results
        belong, which together form a tuple.
    """
    results = []
    for train_data, test_data in dataset.kfolds(k=k):
        model.train(model.preprocess(train_data.standard_df))
        cur_cate = model.predict_cate(test_data.standard_df)
        results.append((cur_cate, test_data.idx))
    return results


def get_cate_subgroups(model, dataset, true_cate=None):
    """Calculate CATE per subgroup"""
    # Train model
    model.train(model.preprocess(dataset.standard_df))

    # Get observed cate
    data = dataset.standard_df.copy(deep=True)
    data["age"] = (data["age"]/10.).astype(int)
    data_subgroups = data[data["nudge"] == 1].groupby(
        model.predictors, as_index=False)["outcome"].mean()
    data_subgroups["cate_obs"] = data_subgroups["outcome"] - data[data["nudge"] == 0].groupby(
        model.predictors, as_index=False)["outcome"].mean()["outcome"]
    data_subgroups["count_nudge"] = data[data["nudge"] == 1].groupby(
        model.predictors, as_index=False)["outcome"].size()["size"]
    data_subgroups["count_control"] = data[data["nudge"] == 0].groupby(
        model.predictors, as_index=False)["outcome"].size()["size"]

    # get subgroups in terms of model.predictors
    subgroups = dataset.standard_df[model.predictors].drop_duplicates().sort_values(
        by=model.predictors, ignore_index=True)

    # Calculate nudge effectiveness for all subgroups
    proba = subgroups.assign(
        probability=model.predict_cate(subgroups))

    # Use age in decades for subgroups
    proba["age"] = (proba["age"]/10.).astype(int)
    prob = proba.groupby(model.predictors, as_index=False)["probability"].mean()
    prob["count"] = proba.groupby(
        model.predictors, as_index=False)["probability"].size()["size"]

    prob = prob.sort_values(by=model.predictors, ignore_index=True)

    cate = "cate_obs"
    if true_cate is not None:
        cate = "cate_model"
        data["cate"] = true_cate
        data_subgroups["cate_model"] = data[data["nudge"] == 1].groupby(
            model.predictors, as_index=False)["cate"].mean()["cate"]

    merged = pd.merge(prob, data_subgroups).drop(columns=['outcome'])
    # print("merged\n", merged.to_string())
    # Only keep subgroups with more than 10 subjects
    merged = merged[(merged["count_nudge"] > 10) & (merged["count_control"] > 10)]

    # Get correlation nudge effectiveness and cate
    result = merged.drop(
        columns=["count_nudge", "count_control"]).corr(method='spearman', min_periods=1)
    # print("Corelation matrix:\n", result)
    # print("correlation cate_obs", result["cate_obs"]["probability"])

    return result[cate]["probability"]


def get_cate_correlations(model, dataset, k=10, ntimes=10):
    """Compute the correlations of the CATE to its modelled estimate

    This only works for simulated datasets, because the true CATE must
    be known.

    Arguments
    ---------
    model: BaseModel
        Model to train and validate with.
    dataset: BaseDataSet
        Dataset to train and validate on.
    k: int
        Number of folds.
    ntimes: int
        Number of times to perform k-fold validation.

    Returns
    -------
    results: list[float]
        List of spearman correlation coefficients for each of the folds
        and repeats.
    """
    all_correlations = []
    if np.all(dataset.cate == dataset.cate[0]):
        return np.zeros(len(k*ntimes))

    for _ in range(ntimes):
        cate_results = get_cate(model, dataset, k=k)
        for res, idx in cate_results:
            if np.all(res == res[0]):
                all_correlations.append(0)
            else:
                all_correlations.append((spearmanr(res, dataset.cate[idx])).correlation)
    return all_correlations


def _get_spearmanr(pred_cate, real_cate):
    if np.all(pred_cate == pred_cate[0]) or np.all(real_cate == real_cate[0]):
        return 0
    return spearmanr(pred_cate, real_cate).correlation


def measure_top_perf(pred_cate, true_cate, frac_select=0.25):
    """Compute the performance by considering the top x%"""
    n_select = round(frac_select*len(pred_cate))
    cate_order = np.argsort(-pred_cate)
    select_idx = cate_order[:n_select]
    best_order = np.argsort(-true_cate)
    best_idx = best_order[:n_select]
    worst_idx = best_order[-n_select:]
    max_perf = np.mean(true_cate[best_idx])
    min_perf = np.mean(true_cate[worst_idx])
    cur_perf = np.mean(true_cate[select_idx])
    return (cur_perf-min_perf)/(max_perf-min_perf)


def get_cate_top_performance(model, dataset, k=5, ntimes=1, frac_select=0.25,
                             spearmanr_results=False):
    """Compute the performance of a model for a given dataset using the top x%
    """
    top_perf = []
    spear_perf = []
    for _ in range(ntimes):
        cate_results = get_cate(model, dataset, k=k)
        for pred_cate, idx in cate_results:
            top_res = measure_top_perf(pred_cate, dataset.cate[idx], frac_select)
            top_perf.append(top_res)
            if spearmanr_results:
                sp_res = _get_spearmanr(pred_cate, dataset.cate[idx])
                spear_perf.append(sp_res)

    if spearmanr_results:
        return np.mean(top_perf), np.mean(spear_perf)
    return np.mean(top_perf)
