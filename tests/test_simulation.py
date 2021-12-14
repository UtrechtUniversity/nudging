import numpy as np
import pytest
from nudging.simulation.utils import create_corr_matrix, mixed_features
from sklearn.linear_model._bayes import BayesianRidge
from nudging.model.biregressor import BiRegressor
from nudging.simulation.multidata import generate_multi_dataset,\
    generate_layered_dataset
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


@pytest.mark.parametrize("control_unique", [0, np.array([0.3, 0.4])])
@pytest.mark.parametrize("control_precision", [0.1, np.array([0.3, 0.4])])
@pytest.mark.parametrize("noise_frac", [0.2, np.array([0.3, 0.4])])
@pytest.mark.parametrize("n_samples", [None, 10000, np.array([300, 3000])])
@pytest.mark.parametrize("linear", [True, False])
@pytest.mark.parametrize("n_dataset", [3, 5])
@pytest.mark.parametrize("layered", [True, False])
def test_multi_dataset(control_unique, control_precision, noise_frac, linear,
                       n_samples, n_dataset, layered):
    if layered:
        generator = generate_layered_dataset
    else:
        generator = generate_multi_dataset
    datasets = generator(
        n_dataset=n_dataset,
        n_features_uncorrelated=10, n_features_correlated=10,
        control_unique=control_unique, control_precision=control_precision,
        noise_frac=noise_frac, linear=linear, n_samples=n_samples)
    assert len(datasets) == n_dataset
    assert np.all([isinstance(d, BaseDataSet) for d in datasets])
    if n_samples is None:
        sample_bounds = [500, 5000]
    elif isinstance(n_samples, np.ndarray):
        sample_bounds = n_samples
    else:
        sample_bounds = [n_samples, n_samples+1]
    assert np.all([len(d) >= sample_bounds[0] and len(d) < sample_bounds[1] for d in datasets])
    if isinstance(control_unique, np.ndarray):
        assert np.all([control_unique[0] <= d.truth["control_unique"] <= control_unique[1]
                       for d in datasets])
    assert np.all([len(d.cate) == len(d) for d in datasets])
