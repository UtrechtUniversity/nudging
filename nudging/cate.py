import numpy as np
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
        model.fit_nudge(data_train["X"], data_train["outcome"],
                        data_train["nudge"])

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


def test_cate(dataset, model, k=30, true_cate=None):
    cate, avg = get_cate(dataset, model, k=k, return_avg=True)
    nudge, outcome = dataset.nudge, dataset.outcome
    nudge_idx = np.where(nudge == 1)[0]
    control_idx = np.where(nudge == 0)[0]
    print(np.mean(outcome[nudge_idx]), np.mean(outcome[control_idx]))
    avg_effect = np.sum(nudge*outcome)/np.sum(nudge) - np.sum((1-nudge)*outcome)/np.sum(1-nudge)
    print(avg_effect, np.mean(outcome), np.sum(nudge*outcome)/np.sum(nudge), np.sum((1-nudge)*outcome)/np.sum(1-nudge))
    normalized_cate = cate - np.mean(cate)
    normalized_outcome = outcome - 0.5*avg_effect*nudge + 0.5*avg_effect*(1-nudge)
    print(np.mean(normalized_outcome[nudge_idx]), np.mean(normalized_outcome[control_idx]))
    print("no cate", pearsonr(avg, normalized_outcome)[0])
    print("cate", pearsonr(avg + 0.5*(2*nudge-1)*normalized_cate, normalized_outcome)[0])
    if true_cate is not None:
        from matplotlib import pyplot as plt
        print(spearmanr(cate, true_cate))
        print(avg.shape, normalized_outcome.shape)
#         print("dot", np.dot(avg-normalized_outcome, -0.5*(2*nudge-1)*normalized_cate))
        print("dot", compute_rank_corr(avg, outcome, nudge, normalized_cate))
        print("dot_true", compute_rank_corr(avg, outcome, nudge, true_cate-np.mean(true_cate)))
        boot_dots = []
        for _ in range(100):
            boot_dots.append(compute_rank_corr(avg, outcome, nudge, normalized_cate[np.random.permutation(len(normalized_cate))]))
#             boot_dots.append(np.dot(avg-normalized_outcome, -0.5*(2*nudge-1)*normalized_cate[np.random.permutation(len(normalized_cate))]))
        print("boot", np.std(boot_dots), np.mean(boot_dots))
        print("mean", np.mean(cate), np.mean(true_cate), np.mean(normalized_cate))
        plt.scatter(normalized_outcome, avg)
        plt.scatter(normalized_outcome, avg + 0.5*(2*nudge-1)*normalized_cate)
        plt.show()
        plt.scatter(true_cate, cate)
        plt.plot(cate, cate)
        plt.scatter(true_cate, normalized_cate)
        plt.show()
        print(pearsonr(cate, true_cate))
        print(linregress(cate, true_cate))
        plt.show()
    print("---------------------------\n\n\n")


def compute_rank_corr(avg, outcome, nudge, normalized_cate):
    train_order = np.argsort(avg)
    train_rank = np.empty_like(train_order)
    train_rank[train_order] = np.arange(len(train_order))

    outcome_order = np.argsort(outcome)
    outcome_rank = np.empty_like(outcome_order)
    outcome_rank[outcome_order] = np.arange(len(outcome_order))

    diff_rank = outcome_rank-train_rank
    return np.dot(diff_rank, 0.5*(2*nudge-1)*normalized_cate)
