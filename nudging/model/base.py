""" Model classes"""
from abc import ABC


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
