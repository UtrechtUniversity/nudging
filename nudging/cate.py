"""Module for calculating conditional average treatment effect (cate) """
import pandas as pd
from scipy.stats import spearmanr
import numpy as np


def get_cate(dataset, model, k=10):
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
    for data_train, data_test in dataset.kfolds(k=k):
        data = pd.concat([data_train["X"], data_train["outcome"],
                          data_train["nudge"]], axis=1)
        model.train(model.preprocess(data))
        cur_cate = model.predict_cate(data_test["X"])
        results.append((cur_cate, data_test["idx"]))
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


def get_cate_correlations(model, dataset, true_cate, k=10, ntimes=10):
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
    for _ in range(ntimes):
        cate_results = get_cate(dataset, model, k=k)
        new_correlations = [spearmanr(x[0], true_cate[x[1]]).correlation
                            for x in cate_results]
        all_correlations.extend(new_correlations)
    return all_correlations
