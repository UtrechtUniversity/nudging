"""Get correlation between predicted cate/probabilty and observed/modelled cate"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
from tqdm import tqdm
import yaml

from nudging.model.base import BaseBiRegressor
from nudging.model.probmodel import ProbModel
from nudging.cate import get_cate_correlations, get_cate_subgroups
from nudging.simulation.multidata import generate_multi_dataset

def equal_range(start, stop, n_step):
    if n_step >= stop-start:
        for i in range(start, stop):
            yield(range(i, i+1))
        return range(start, stop)
    step_size = (stop-start)//n_step
    remainder = (stop-start) - step_size*n_step
    cur_start = 0
    for i in range(n_step):
        cur_stop = cur_start + int(i < remainder) + step_size
        yield(range(cur_start, cur_stop))
        cur_start = cur_stop


def smooth_data(x, y, n_data=100):
    x_sorted = np.argsort(x)
    new_x = []
    new_y = []
    for i_data in equal_range(0, len(x), n_data):
        new_x.append(np.mean(x[x_sorted[i_data]]))
        new_y.append(np.mean(y[x_sorted[i_data]]))
    return new_x, new_y

def plot_correlations(datasets, attr, *args, **kwargs):
    
    plt.figure(dpi=120)
    for n, arg in enumerate(args):
        x = np.array([data.truth[attr] for data in datasets])
        # xcor = spearmanr(x, correlations).correlation
        plt.plot(*smooth_data(x, np.array(arg), **kwargs), label=n)

    plt.legend()
    plt.xlabel(attr)
    plt.ylabel("correlation")
    # plt.title(xcor)
    plt.savefig(attr+'.png')


# Generate simulated datasets
np.random.seed(9817274)
datasets = generate_multi_dataset(
    n_dataset=1000,
    n_nudge_type=1,
    n_nudge_domain=1,
    dataset_weight=1,
    eigen_power=2.,
    n_features_correlated=2,
    n_features_uncorrelated=2)


# Choose model
model1 = BaseBiRegressor(BayesianRidge())
model2 = ProbModel(LogisticRegression())

# Get predictors from config.yaml
config = yaml.safe_load(open("config.yaml"))
model1.predictors = config["features"]
model2.predictors = config["features"]

# Compute correla
correlations1 = []
correlations2 = []

# Plot correlation for subgroups if True (observed), else individuals (modelled)
subgroups = True

for d in tqdm(datasets):

    if subgroups:
        # observed cate per subgroup
        cor = get_cate_subgroups(d, model1)
        correlations1.append(cor)
        cor = get_cate_subgroups(d, model2)
        correlations2.append(cor)
    else:
        cor = get_cate_correlations(d, d.truth["cate"], model1)    
        correlations1.append(np.mean(cor))
        cor = get_cate_correlations(d, d.truth["cate"], model2)    
        correlations2.append(np.mean(cor))

plot_correlations(datasets, "nudge_avg",  correlations1, correlations2, n_data=40)
plot_correlations(datasets, "noise_frac", correlations1, correlations2)
plot_correlations(datasets, "n_samples", correlations1, correlations2)
plot_correlations(datasets, "control_unique", correlations1, correlations2, n_data=70)
plot_correlations(datasets, "control_precision", correlations1, correlations2, n_data=50)



