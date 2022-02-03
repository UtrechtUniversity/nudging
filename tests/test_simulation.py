import numpy as np
import pytest
from nudging.simulation.utils import create_corr_matrix, mixed_features
from sklearn.linear_model._bayes import BayesianRidge
from nudging.model.biregressor import BiRegressor
from nudging.simulation import generate_datasets
from nudging.dataset.base import BaseDataSet


@pytest.mark.parametrize("n_features_uncorrelated", [1, 8])
@pytest.mark.parametrize("n_features_correlated", [1, 10])
@pytest.mark.parametrize("eigen_power", [-0.3, 1, 3])
def test_create_corr_matrix(n_features_uncorrelated, n_features_correlated, eigen_power):
    corr_matrix = create_corr_matrix(n_features_uncorrelated, n_features_correlated, eigen_power)
    n_features = n_features_uncorrelated + n_features_correlated + 2
    assert corr_matrix.shape == (n_features, n_features)
    assert np.all(np.linalg.eigvals(corr_matrix) > 0)
    assert np.allclose(np.diag(corr_matrix), 1)
    uncor = corr_matrix[:n_features_uncorrelated, :n_features_uncorrelated]
    assert np.allclose(uncor, np.identity(uncor.shape[0]))
    assert np.sum(np.isclose(corr_matrix, 0)) == n_features_uncorrelated**2 - n_features_uncorrelated


@pytest.mark.parametrize("control_unique", [0, 0.5])
@pytest.mark.parametrize("control_precision", [0.1, 0.6])
@pytest.mark.parametrize("noise_frac", [0.2, 0.8])
@pytest.mark.parametrize("n_samples", [10000, 20000])
@pytest.mark.parametrize("linear", [True, False])
def test_mixed_features(control_unique, control_precision, noise_frac, linear, n_samples):
    fm = mixed_features(n_features_uncorrelated=10,
                        n_features_correlated=10,
                        control_unique=control_unique,
                        control_precision=control_precision,
                        noise_frac=noise_frac, linear=linear,
                        n_samples=n_samples)
    assert len(fm) == n_samples
    model = BiRegressor(BayesianRidge())
    X, nudge, outcome = model._X_nudge_outcome(fm.standard_df)
    assert X.shape == (n_samples, 20)
    assert len(nudge) == n_samples
    assert len(outcome) == n_samples


def check_datasets(**kwargs):
    datasets = generate_datasets(**kwargs)
    assert len(datasets) == kwargs["n"]
    for d in datasets:
        for param_name, param_val in kwargs.items():
            if param_name == "n" or param_name == "n_layers" or param_name == "n_rescale":
                continue
            true_val = d.truth[param_name]
            if isinstance(param_val, (np.ndarray, tuple)):
                assert true_val >= param_val[0] and true_val <= param_val[1]
            else:
                assert np.fabs(true_val - param_val) < 1e-7, param_name
    assert np.all([isinstance(d, BaseDataSet) for d in datasets])
    assert np.all([len(d) == d.truth["n_samples"] for d in datasets])
    assert np.all([len(d.cate) == len(d) for d in datasets])


def test_multi_dataset():
    param_dict = {
        "noise_frac": [0.2, np.array([0.3, 0.4])],
        "control_unique": [0, np.array([0.3, 0.4])],
        "control_precision": [0.1, np.array([0.3, 0.4])],
        "n_samples": [2000, np.array([300, 3000])],
        "linear": [True, False],
        "n_samples": [1000, np.array([300, 3000])],
        "avg_correlation": [0.05, np.array([0.05, 0.1])],
        "n_features_uncorrelated": [7, (1, 5)],
        "n_features_correlated": [7, (1, 5)],
        "nudge_avg": [0.3, (-0.1, 0.1)],
        "n_rescale": [1, (0, 3)],
        "n_layers": [1, (2, 10)],
    }
    for _ in range(10):
        n = np.random.randint(1, 5)
        kwargs = {}
        for param_name, param_vals in param_dict.items():
            n_vals = len(param_vals)
            rand = np.random.randint(0, n_vals+1)
            if rand == n_vals:
                continue
            kwargs[param_name] = param_vals[rand]
        check_datasets(n=n, **kwargs)
