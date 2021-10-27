import numpy as np
from nudging.simulation.multidata import generate_multi_dataset
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
import yaml

from nudging.model.base import BaseBiRegressor
from nudging.model.probmodel import ProbModel
from nudging.cate import get_cate, get_cate_correlations
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from tqdm import tqdm


# Generate simulated datasets
np.random.seed(9817274)
datasets = generate_multi_dataset(1000, n_nudge_type=1, n_nudge_domain=1, dataset_weight=1)

# Choose model
model = BaseBiRegressor(BayesianRidge())
# model = ProbModel(LogisticRegression())

# Get predictors from config.yaml
config = yaml.safe_load(open("config.yaml"))
model.predictors = config["features"]

# Compute correla
correlations = []
std_correlations = []
for d in tqdm(datasets):
    # print(d.obs_cate(model.predictors))
    cor = get_cate_correlations(d, d.truth["cate"], model)
    correlations.append(np.mean(cor))
    std_correlations.append(np.std(cor))

plt.scatter(correlations, std_correlations)
plt.savefig("correlation.png")
plt.show()

