"""Regression model class"""
import numpy as np
from sklearn.base import clone

from nudging.model.base import BaseModel


class BaseBiRegressor(BaseModel):
    """Base class for BiRegressor model"""
    def __init__(self, model, predictors=None):
        super().__init__(model)
        self.model_nudge = model
        self.model_control = clone(model)
        self.predictors = predictors

    def train(self, data):
        # Drop rows with missing values for predictors
        if self.predictors is None:
            self.predictors = [x for x in list(data) if x not in ["nudge", "outcome"]]
        nudge_idx = (data["nudge"] == 1)
        control_idx = (data["nudge"] == 0)
        self.model_nudge.fit(data[self.predictors][nudge_idx], data["outcome"][nudge_idx])
        self.model_control.fit(data[self.predictors][control_idx], data["outcome"][control_idx])

    def predict_outcome(self, data):
        nudge_idx = (data["nudge"] == 1)
        control_idx = (data["nudge"] == 0)
        result = np.zeros(len(data["nudge"]))
        if nudge_idx.any():
            result[nudge_idx] = self.model_nudge.predict(data[self.predictors][nudge_idx])
        if control_idx.any():
            result[control_idx] = self.model_control.predict(data[self.predictors][control_idx])

        return result

    def clone(self):
        return BaseBiRegressor(clone(self.model_nudge),
                               predictors=self.predictors)
