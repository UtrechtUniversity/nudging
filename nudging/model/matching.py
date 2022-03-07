from nudging.model.base import BaseModel
from sklearn.base import clone
from nudging.model.biregressor import BiRegressor


class MatchingModel(BaseModel):
    def __init__(self, model, predictors=None, tlearner=True):
        super().__init__(model, predictors)
        self.nudge_model = clone(model)
        self.control_model = clone(model)
        self.train_tlearner = tlearner
        if self.train_tlearner:
            self.tlearner = BiRegressor(clone(model), self.predictors)

    def predict_outcome(self, data):
        X, nudge = self._X_nudge_transform(data)
        if not self.tlearner:
            return (nudge-0.5)*self.model.predict(X)
        return self.tlearner._predict(X, nudge)
        # nudge_idx = np.where(nudge == 1)
        # control_idx = np.where(nudge == 0)
        # result = np.zeros_like(nudge)
        # result[nudge_idx] = self.nudge_model.predict(X[nudge_idx])
        # result[control_idx] = self.control_model.predict(X[control_idx])
        # return result
