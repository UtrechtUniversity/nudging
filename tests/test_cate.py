import pytest
import numpy as np
from nudging.simulation.utils import mixed_features
from nudging.model.biregressor import BiRegressor
from sklearn.linear_model._bayes import BayesianRidge
from nudging.cate import get_cate, get_cate_subgroups, get_cate_correlations,\
    measure_top_perf, get_cate_top_performance
from nudging.simulation import generate_multi_dataset


@pytest.fixture
def sim_matrix():
    np.random.seed(123474)
    return mixed_features(noise_frac=0.0)


@pytest.fixture
def sim_matrix_age():
    np.random.seed(1298745)
    return generate_multi_dataset(1)[0]


def test_get_cate(sim_matrix):
    model = BiRegressor(BayesianRidge())
    results = get_cate(model, sim_matrix, k=10)
    assert len(results) == 10
    all_idx = []
    for res in results:
        all_idx.extend(res[1])
    assert len(all_idx) == len(sim_matrix)
    assert len(all_idx) == len(np.unique(all_idx))


def test_cate_subgroups(sim_matrix_age):
    model = BiRegressor(BayesianRidge())
    model.predictors = ["age", "gender"]
    results = get_cate_subgroups(model, sim_matrix_age, true_cate=sim_matrix_age.cate)
    assert not np.isnan(results)


def test_get_cate_correlations(sim_matrix):
    model = BiRegressor(BayesianRidge())
    results = get_cate_correlations(model, sim_matrix, k=10, ntimes=10)
    assert len(results) == 100
    assert np.mean(results) > 0


def test_top_perf(sim_matrix):
    true_cate = np.random.randn(100)
    cate_order = np.argsort(true_cate)
    n_select = 25
    reverse_cate = np.ones_like(true_cate)
    reverse_cate[cate_order[-n_select:]] = 0 
    assert np.isclose(measure_top_perf(true_cate, true_cate), 1)
    assert np.isclose(measure_top_perf(true_cate, 3*true_cate), 1)
    assert np.isclose(measure_top_perf(1-reverse_cate, reverse_cate), 0)
    model = BiRegressor(BayesianRidge())
    assert get_cate_top_performance(model, sim_matrix) > 0.5
