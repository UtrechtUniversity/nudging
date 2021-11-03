""" Model classes"""
from abc import ABC
import numpy as np

from sklearn.base import clone

class BaseModel(ABC):
    """Base class for model"""

    predictors = None

    def __init__(self, model):
        self.model = model

    @staticmethod
    def preprocess(data_frame):
        """Standardize outcome"""
        return data_frame

    def train(self, data):
        """Train model"""
        # Drop rows with missing values for predictors
        df_nonan = data.dropna(subset=self.predictors, inplace=False)
        self.model.fit(
            df_nonan[self.predictors],
            df_nonan["outcome"]
        )

    def predict_outcome(self, data):
        """Predict outcome"""
        return self.model.predict_proba(data[self.predictors])[:, 1]

    def predict_cate(self, data):
        """Predict conditional averagte treatment effect"""
        nudge_data = data[self.predictors].copy(deep=True)
        nudge_data["nudge"] = 1
        control_data = data[self.predictors].copy(deep=True)
        control_data["nudge"] = 0
        nudge_pred = self.predict_outcome(nudge_data)
        control_pred = self.predict_outcome(control_data)
        cate = nudge_pred - control_pred
        return cate


class BaseBiRegressor(BaseModel):

    """Base class for BiRegressor model"""
    def __init__(self, model):
        super().__init__(model)
        self.model_nudge = model
        self.model_control = clone(model)

    def train(self, data):
        # Drop rows with missing values for predictors
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
