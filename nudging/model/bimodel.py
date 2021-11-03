"""Bimodel"""
from collections import defaultdict

import numpy as np
from scipy.optimize import bisect
from sklearn.svm import SVC

from nudge.test.prior import compute_priors
from sklearn.linear_model._base import LinearRegression
from scipy.stats.stats import spearmanr


class BiModel():
    def __init__(self, *model_args, model_class=SVC, n_val=100, **model_kwargs):
        self._model_class = model_class
        self._model_args = model_args
        self._model_kwargs = model_kwargs
        self._model_kwargs["probability"] = True
        self._n_val = n_val

    def fit_nudge(self, X, outcome, nudge):
        nudge_idx = np.where(nudge == 1)[0]
        control_idx = np.where(nudge == 0)[0]
        X_nudge = np.hstack((X, nudge.reshape(-1, 1)))
        nudge_pred, self.nudge_models = self._generate_models(
            X[nudge_idx], outcome[nudge_idx], n_val=self._n_val)
        control_pred, self.control_models = self._generate_models(
            X[control_idx], outcome[control_idx], n_val=self._n_val)
        all_pred, all_models = self._generate_models(
            X_nudge, outcome, n_val=self._n_val)
#         all_model = self._model_class(*self._model_args, **self._model_kwargs, probability=True)
#         all_X =
#         all_model.fit(all_X, compute_y(outcome)[0])
        base_prediction = np.zeros(len(outcome))
        base_prediction[control_idx] = control_pred
        base_prediction[nudge_idx] = nudge_pred
        print("x", spearmanr(base_prediction, outcome))
        print("y", spearmanr(nudge_pred, outcome[nudge_idx]))
        print("x", spearmanr(base_prediction[nudge_idx], outcome[nudge_idx]))
        print("z", spearmanr(control_pred, outcome[control_idx]))
        self.nudge_f = model_transformation(nudge_pred, outcome[nudge_idx])
        self.control_f = model_transformation(control_pred, outcome[control_idx])
#         self.control_models = control_models
#         self.nudge_models = nudge_models
#         self.offset = compute_offset(base_prediction, nudge, outcome, n_val=self._n_val)
#         self.nudge_f, self.control_f = combine_models(nudge_pred, control_pred, outcome, nudge)
#         print(self.offset)

    def predict_nudge(self, X, nudge):
        nudge_idx = np.where(nudge == 1)[0]
        control_idx = np.where(nudge == 0)[0]
        nudge_prob = []
        control_prob = []
        if len(nudge_idx) > 0:
            nudge_prob = compute_probabilities(
                X[nudge_idx], self.nudge_models, transform=self.nudge_f)
        if len(control_idx) > 0:
            control_prob = compute_probabilities(
                X[control_idx], self.control_models, transform=self.control_f)
        proba = np.zeros(len(nudge))
        proba[nudge_idx] = nudge_prob
        proba[control_idx] = control_prob
        proba = proba.reshape(-1, 1)
        return np.hstack((1-proba, proba))

    def generate_combined_model(self, X, outcome, n_val=100):
        pass

    def _generate_models(self, X, outcome, n_val=100):
        results = defaultdict(lambda: [])
        models = []
#         for p_one in np.linspace(0.3, 0.7, n_val):
        p_one = 0.5
        for _ in range(n_val):
            y, train_idx, val_idx = train_test_split(X, outcome, p_one=p_one)
            model = self.new_model
            model.fit(X[train_idx], y[train_idx])
            models.append(model)
            y_pred_val = model.predict_proba(X[val_idx])[:, 1]
            for j, val_id in enumerate(val_idx):
                results[val_id].append(y_pred_val[j])
        avg_results = np.zeros(len(outcome))
        for val_id, res in results.items():
            avg_results[val_id] = np.mean(res)
        return avg_results, models

    @property
    def new_model(self):
        return self._model_class(*self._model_args, **self._model_kwargs)


class MultiModel():
    def __init__(self):
        pass


def compute_probabilities(X, models, transform=None):
    results = np.zeros(X.shape[0])
#     n_val = len(models)
    for model in models:
        results += model.predict_proba(X)[:, 1]
    results /= len(models)
    if transform is None:
        return results
    return transform(results)
#     return offset[0] + offset[1]*results/len(models) + offset[2]
#     results /= n_val
#     results[results == 0] = 0.5/n_val
#     results[results == 1] = (n_val-0.5)/n_val
#     df_results = -np.log(1/results-1)
#     df_results += offset
#     pval = 1/(1+np.exp(-df_results))
#     return pval


# def compute_probabilities(X, models, offset):
#     results = np.zeros(X.shape[0])
#     n_val = len(models)
#     for model in models:
#         results += model.predict(X)
#     results /= n_val
#     results[results == 0] = 0.5/n_val
#     results[results == 1] = (n_val-0.5)/n_val
#     df_results = -np.log(1/results-1)
#     df_results += offset
#     pval = 1/(1+np.exp(-df_results))
#     return pval
#
#
# def compute_offset(base_prediction, nudge, outcome, n_val):
#     target_prior_matrix = compute_priors(nudge, outcome)
#     target_ratio = target_prior_matrix[0]
#     df_base = base_prediction.copy()
#     df_base[np.where(base_prediction == 0)[0]] = 0.5/n_val
#     df_base[np.where(base_prediction == 1)[0]] = (n_val-0.5)/n_val
#     df_base = -np.log(1/df_base-1)
#     bounds = [df_base.min() - df_base.max(), df_base.max()-df_base.min()]
#
#     def compute_diff(param):
#         test_outcome = base_prediction+nudge*param
#         new_prior_matrix = compute_priors(nudge, test_outcome)
#         new_ratio = new_prior_matrix[0]
#         return target_ratio-new_ratio
#     return bisect(compute_diff, *bounds)


def compute_offset(base_prediction, nudge, outcome, n_val):
    nudge_one_idx = np.where(nudge == 1)
    nudge_zero_idx = np.where(nudge == 0)
    model = LinearRegression()
    model.fit(base_prediction[nudge_one_idx].reshape(-1, 1), outcome[nudge_one_idx])
    inter_one, coef_one = model.intercept_, model.coef_[0]
    model = LinearRegression()
    model.fit(base_prediction[nudge_zero_idx].reshape(-1, 1), outcome[nudge_zero_idx])
    inter_zero, coef_zero = model.intercept_, model.coef_[0]
    second_prediction = np.zeros_like(base_prediction)
    second_prediction[nudge_one_idx] = inter_one+coef_one*base_prediction[nudge_one_idx]
    second_prediction[nudge_zero_idx] = inter_zero+coef_zero*base_prediction[nudge_zero_idx]
    target_prior_matrix = compute_priors(nudge, outcome)
    target_ratio = target_prior_matrix[0]
#     df_base = base_prediction.copy()
    p_min = second_prediction.min()
    p_max = second_prediction.max()

    bounds = [p_min-p_max, p_max-p_min]

#     from matplotlib import pyplot as plt
#     plt.scatter(outcome, second_prediction)
#     plt.plot([0, 1], [0, 1])
#     plt.show()

    def compute_diff(offset):
        test_outcome = second_prediction+nudge*offset
        new_prior_matrix = compute_priors(nudge, test_outcome)
        new_ratio = new_prior_matrix[0]
        return target_ratio-new_ratio
#     print(inter_one, coef_one, bisect(compute_diff, *bounds), inter_zero, coef_zero, 0)
    return inter_one, coef_one, bisect(compute_diff, *bounds), inter_zero, coef_zero, 0


def model_transformation(prediction, outcome):
#     nudge_idx = np.where(nudge == 1)[0]
#     control_idx = np.where(nudge == 0)[0]
#     print(spearmanr(nudge_pred, all_pred[nudge_idx]))
#     print(spearmanr(control_pred, all_pred[control_idx]))
#
#     def compute_trans(idx, pred, all_pred):
    print(spearmanr(prediction, outcome).correlation)

    def f(y):
        y_new = np.copy(y)
        y_new -= np.mean(prediction)
        y_new /= np.std(prediction)
        if spearmanr(prediction, outcome).correlation < 0:
            y_new *= -1
        y_new *= np.std(outcome)
        y_new += np.mean(outcome)
        return y_new
    return f
#     return (compute_trans(nudge_idx, nudge_pred, all_pred),
#             compute_trans(control_idx, control_pred, all_pred))
#     control_trans = []
#     nudge_mean_shift =  -
#     pass

def train_test_split(X, outcome, p_one=0.5, train_size=0.9):
#     break_value = np.quantile(outcome, p_one)
#     zero_idx = np.where(outcome < break_value)[0]
#     one_idx = np.where(outcome > break_value)[0]
#     maybe_idx = np.where(outcome == break_value)[0]
#     n_choice = round(len(outcome)*p_one)-len(one_idx)
#     n_choice = max(0, min(n_choice, len(maybe_idx)))
#     extra_one_idx = np.random.choice(len(maybe_idx), size=n_choice, replace=False)
#     one_idx = np.append(maybe_idx[extra_one_idx], one_idx)
#     zero_idx = np.append(maybe_idx[np.delete(np.arange(len(maybe_idx)), extra_one_idx)], zero_idx)
#     y = np.zeros(len(outcome), dtype=int)
#     y[one_idx] = 1

    y, zero_idx, one_idx = compute_y(outcome, p_one)
    train_one_idx = np.random.choice(one_idx, size=round(train_size*len(one_idx)), replace=False)
    train_zero_idx = np.random.choice(zero_idx, size=round(train_size*len(zero_idx)), replace=False)
    train_idx = np.append(train_one_idx, train_zero_idx)
    val_idx = np.delete(np.arange(len(y)), train_idx)
    return y, np.sort(train_idx), val_idx


def compute_y(outcome, p_one=0.5):
    break_value = np.quantile(outcome, p_one)
    zero_idx = np.where(outcome < break_value)[0]
    one_idx = np.where(outcome > break_value)[0]
    maybe_idx = np.where(outcome == break_value)[0]
    n_choice = round(len(outcome)*p_one)-len(one_idx)
    n_choice = max(0, min(n_choice, len(maybe_idx)))
    extra_one_idx = np.random.choice(len(maybe_idx), size=n_choice, replace=False)
    one_idx = np.append(maybe_idx[extra_one_idx], one_idx)
    zero_idx = np.append(maybe_idx[np.delete(np.arange(len(maybe_idx)), extra_one_idx)], zero_idx)
    y = np.zeros(len(outcome), dtype=int)
    y[one_idx] = 1
    return y, zero_idx, one_idx
