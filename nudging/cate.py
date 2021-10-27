import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.stats.mstats_basic import linregress


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
        data = pd.concat([data_train["X"], data_train["outcome"], data_train["nudge"]], axis=1)
        model.train(model.preprocess(data))
        cur_cate = model.predict_cate(data_test["X"])
        results.append((cur_cate, data_test["idx"]))
    return results


def get_cate_correlations(dataset, true_cate, model, k=10, n=10):
    """Compute the correlations of the CATE to its modelled estimate

    This only works for simulated datasets, because the true CATE must
    be known.

    Arguments
    ---------
    dataset: BaseDataSet
        Dataset to train and validate on.
    model: BaseModel
        Model to train and validate with.
    k: int
        Number of folds.
    n: int
        Number of times to perform k-fold validation.

    Returns
    -------
    results: list[float]
        List of spearman correlation coefficients for each of the folds
        and repeats.
    """
    # true_cate = dataset.truth["cate"]
    all_correlations = []
    for _ in range(n):
        cate_results = get_cate(dataset, model, k=k)
        new_correlations = [spearmanr(x[0], true_cate[x[1]]).correlation
                            for x in cate_results]
        all_correlations.extend(new_correlations)
    return all_correlations

