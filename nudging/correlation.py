"""Plot correlation between predicted cate/probabilty and observed/modelled cate"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import yaml

from nudging.model.biregressor import BiRegressor
from nudging.model.probmodel import ProbModel
from nudging.cate import get_cate_correlations, get_cate_subgroups
from nudging.simulation.multidata import generate_multi_dataset


def equal_range(start, stop, n_step):
    """Get n_step equal ranges between start and stop
    Args:
        start (int): start of intervals
        stop (int): end of intervals
        n_step (int): number of intervals
    Yields:
        range: series of integer numbers, which we can iterate using a for loop
    """
    if n_step >= stop - start:
        for i in range(start, stop):
            yield range(i, i+1)
        return range(start, stop)
    step_size = (stop-start)//n_step
    remainder = (stop-start) - step_size*n_step
    cur_start = 0
    for i in range(n_step):
        cur_stop = cur_start + int(i < remainder) + step_size
        yield range(cur_start, cur_stop)
        cur_start = cur_stop
    return None


def smooth_data(xdata, ydata, n_data=100):
    """Smooth data between intervals
    Args:
        xdata (ndarray): array with x-axis data
        ydata (ndarray): array with y-axis data
        n_data (int): number of data points in an interval used for smoothing
    Returns:
        tuple: tuple of ndarrays
    """
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    if len(np.unique(xdata)) < n_data:
        new_x = np.unique(xdata)
        new_y = []
        for x in new_x:
            new_y.append(np.mean(ydata[xdata==x]))
        return new_x, new_y
    x_sorted = np.argsort(xdata)
    new_x = []
    new_y = []
    for i_data in equal_range(0, len(xdata), n_data):
        new_x.append(np.nanmean(xdata[x_sorted[i_data]]))
        new_y.append(np.nanmean(ydata[x_sorted[i_data]]))
    return new_x, new_y


def plot_correlations(outdir, datasets_, attr, *args, **kwargs):
    """Plot correlations
    Args:
        outdir (str): folder for saving plots
        datasets_ (nudging.DataSet):
        attr (str): attribute to use as xaxis to plot correlations
        args: sequence of arrays with correlations
        kwargs: keyword arguments to pass to function smooth_data
    """
    plt.figure(dpi=120)
    for num, arg in enumerate(args):
        xdata = np.array([data.truth[attr] for data in datasets_])
        plt.plot(*smooth_data(xdata, np.array(arg), **kwargs), label=num)

    plt.ylim([0, 1.05])
    plt.legend()
    plt.xlabel(attr)
    plt.ylabel("correlation")
    filename = Path(outdir, attr + ".png")
    plt.savefig(filename)


if __name__ == "__main__":

    # Choose model
    model1 = BiRegressor(BayesianRidge())
    model2 = ProbModel(LogisticRegression())

    # Get predictors from config.yaml
    config = yaml.safe_load(open("config.yaml"))
    model1.predictors = config["features"]
    model2.predictors = config["features"]

    # Generate simulated datasets
    np.random.seed(9817274)
    datasets = generate_multi_dataset(
        n_dataset=1000,
        n_nudge_type=1,
        n_nudge_domain=1,
        dataset_weight=1,
        eigen_power=1.5,
        n_features_correlated=2,
        n_features_uncorrelated=2)

    attributes = ["nudge_avg", "noise_frac", "n_samples", "control_unique", "control_precision"]

    # Compute correlations

    correlations1 = []
    correlations2 = []
    # subgroups observed cate
    for d in tqdm(datasets):
        PLOTDIR = "plots_subgroups_cate_obs"
        cor = get_cate_subgroups(model1, d)
        correlations1.append(cor)
        cor = get_cate_subgroups(model2, d)
        correlations2.append(cor)

    for attribute in attributes:
        plot_correlations(PLOTDIR, datasets, attribute,  correlations1, correlations2, n_data=50)

    # subgroups modelled cate
    correlations1 = []
    correlations2 = []
    for d in tqdm(datasets):
        PLOTDIR = "plots_subgroups_cate_model"
        cor = get_cate_subgroups(model1, d, d.truth["cate"])
        correlations1.append(cor)
        cor = get_cate_subgroups(model2, d, d.truth["cate"])
        correlations2.append(cor)

    for attribute in attributes:
        plot_correlations(PLOTDIR, datasets, attribute,  correlations1, correlations2, n_data=50)

    # individual correlations
    correlations1 = []
    correlations2 = []
    for d in tqdm(datasets):
        PLOTDIR = "plots_ind_cate_model"
        cor = get_cate_correlations(model1, d, d.truth["cate"])
        correlations1.append(np.mean(cor))
        cor = get_cate_correlations(model2, d, d.truth["cate"])
        correlations2.append(np.mean(cor))

    for attribute in attributes:
        plot_correlations(PLOTDIR, datasets, attribute,  correlations1, correlations2, n_data=50)
