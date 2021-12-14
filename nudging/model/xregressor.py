import numpy as np
from sklearn.base import clone

from nudging.model.base import BaseModel
from nudging.model.biregressor import BiRegressor


class XRegressor(BaseModel):
    """Class for X-learner regression

    See https://www.pnas.org/cgi/doi/10.1073/pnas.1804597116. It trains
    two BiRegressors (T-learners) and then uses a crossover mechanism.
    """
    def __init__(self, model, predictors=None):
        super().__init__(model, predictors)
        self.biregressor = BiRegressor(model, predictors=predictors)
        self.nudge_control_model = clone(model)
        self.control_nudge_model = clone(model)

    def _fit(self, X, nudge, outcome):
        self.biregressor._fit(X, nudge, outcome)
        nudge_idx = np.where(nudge == 1)[0]
        control_idx = np.where(nudge == 0)[0]
        imputed_treatment_nudge = self.biregressor._predict(
            X[control_idx], 1-nudge[control_idx]) - outcome[control_idx]
        imputed_treatment_control = outcome[nudge_idx] - self.biregressor._predict(
            X[nudge_idx], 1-nudge[nudge_idx])
        self.nudge_control_model.fit(X[nudge_idx], imputed_treatment_control)
        self.control_nudge_model.fit(X[control_idx], imputed_treatment_nudge)

    def train(self, data):
        self._fit(*self._X_nudge_outcome(data))

    def _predict(self, X, nudge):
        control_cate = self.control_nudge_model.predict(X)
        nudge_cate = self.nudge_control_model.predict(X)
        cate = 0.5*(control_cate + nudge_cate)
        base_pred = self.biregressor._predict(X, np.zeros_like(nudge))
        return base_pred + cate*nudge

    def predict_outcome(self, data):
        return self._predict(*self._X_nudge(data))

#
#     def predict_cate(self, X, return_avg=False):
#         control_cate = self.control_nudge_model.predict(X)
#         nudge_cate = self.nudge_control_model.predict(X)
#         cate = 0.5*(control_cate + nudge_cate)
#         if not return_avg:
#             return cate
#         base_pred = self.base_model.predict_nudge(X, np.zeros(len(cate)))
#         return cate, base_pred+0.5*cate
