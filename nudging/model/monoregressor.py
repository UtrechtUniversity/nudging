from nudging.model.base import BaseModel
import numpy as np


class MonoRegressor(BaseModel):
    """Class for MonoRegressor model

    This model is a s-learner, which means that only a single model
    is trained, with the treatment indicator as an extra feature.
    """
    def train(self, data):
        X, nudge, outcome = self._X_nudge_outcome(data)
        X_new = np.hstack((X, nudge.reshape(-1, 1)))
        self.model.fit(X_new, outcome)

    def predict_outcome(self, data):
        X, nudge = self._X_nudge(data)
        X_new = np.hstack((X, nudge.reshape(-1, 1)))
        return self.model.predict(X_new)
