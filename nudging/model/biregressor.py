"""BiRegression model class"""
import numpy as np
from sklearn.base import clone

from nudging.model.base import BaseModel


class BiRegressor(BaseModel):
    """Class for BiRegressor model

    This model is a t-learner, which means that two models
    are trained: one on the nudged group and one on the control
    group. The CATE is then simply computed by the difference
    between the models.

    Arguments
    ---------
    model: sklearn.BaseModel
        Initialized model to train with.
    predictors: list[str]
        List of columns to train/predict model with.
        None means all columns except nudge/outcome.
    """
    def __init__(self, model, predictors=None):
        super().__init__(model)
        self.model_nudge = model
        self.model_control = clone(model)
        self.predictors = predictors

    def _fit(self, X, nudge, outcome):
        nudge_idx = (nudge == 1)
        control_idx = (nudge == 0)
        self.model_nudge.fit(X[nudge_idx], outcome[nudge_idx])
        self.model_control.fit(X[control_idx], outcome[control_idx])

    def train(self, data):
        # Drop rows with missing values for predictors
        if self.predictors is None:
            self.predictors = [x for x in list(data)
                               if x not in ["nudge", "outcome"]]
        self._fit(data[self.predictors].values, data["nudge"].values,
                  data["outcome"].values)

    def _predict(self, X, nudge):
        nudge_idx = (nudge == 1)
        control_idx = (nudge == 0)
        result = np.zeros(len(nudge))
        if nudge_idx.any():
            result[nudge_idx] = self.model_nudge.predict(X[nudge_idx])
        if control_idx.any():
            result[control_idx] = self.model_control.predict(X[control_idx])
        return result

    def predict_outcome(self, data):
        return self._predict(*self._X_nudge(data))

    def clone(self):
        return BiRegressor(clone(self.model_nudge),
                           predictors=self.predictors)
