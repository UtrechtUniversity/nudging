import pytest
from nudging.simulation.utils import mixed_features
from nudging.model.biregressor import BiRegressor
from nudging.model.monoregressor import MonoRegressor
from nudging.model.xregressor import XRegressor
from sklearn.linear_model import BayesianRidge
from nudging.model.base import BaseModel
from scipy.stats.stats import spearmanr
import numpy as np
from nudging.model.probmodel import ProbModel
from sklearn.linear_model import LogisticRegression
from nudging.simulation.multidata import generate_multi_dataset


@pytest.fixture
def sim_matrix():
    np.random.seed(123474)
    return mixed_features(noise_frac=0.0)


@pytest.fixture
def sim_matrix_age_gender():
    np.random.seed(8127384)
    return generate_multi_dataset(1)[0]


def check_prediction(sim_matrix, model, performance_test=True, clone=False):
    assert isinstance(model, BaseModel)
    all_cor_out = []
    all_cor_cate = []
    for train_data, test_data in sim_matrix.kfolds(k=5):
        true_outcome = test_data.outcome
        model.train(model.preprocess(train_data.standard_df))
        test_cate = model.predict_cate(test_data.standard_df)
        test_outcome = model.predict_outcome(test_data.standard_df)

        assert len(test_outcome) == len(test_data)
        assert len(test_cate) == len(test_data)
        all_cor_out.append(spearmanr(test_outcome, true_outcome).correlation)
        all_cor_cate.append(spearmanr(test_cate, test_data.cate).correlation)

    if performance_test:
        assert np.mean(all_cor_out) > 0
        assert np.mean(all_cor_cate) > 0

    assert isinstance(model.clone(), model.__class__)
    if not clone:
        check_prediction(sim_matrix, model, performance_test, clone=True)


@pytest.mark.parametrize("model_type", [BiRegressor, MonoRegressor, XRegressor])
def test_regression_models(sim_matrix, model_type):
    model = model_type(BayesianRidge())
    # Mono regressors are kind of bad, so don't check the performance on those?
    check_prediction(sim_matrix, model, model_type != MonoRegressor)


def test_proba_model(sim_matrix_age_gender):
    model = ProbModel(LogisticRegression())
    model.predictors = ["age", "gender"]
    check_prediction(sim_matrix_age_gender, model)
