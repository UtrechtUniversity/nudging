from sklearn.svm import SVC
from nudge.model.base import BaseModel
import numpy as np
from .bimodel import train_test_split


class SVMModel(BaseModel, SVC):
    def __init__(self, *args, probability=True, **kwargs):
        super().__init__(*args, **kwargs, probability=probability)


class SVMModel2(BaseModel):
    def __init__(self, *args, n_val=100, **kwargs):
        kwargs["probability"] = True
        self._args = args
        self._kwargs = kwargs
        self.n_val = n_val

    def new_model(self):
        return SVC(*self._args, **self._kwargs)

    def fit(self, X, outcome):
        models = []
        for p_one in np.linspace(0.3, 0.7, self.n_val):
            y, train_idx, val_idx = train_test_split(X, outcome, p_one=p_one)
            model = self.new_model()
            model.fit(X[train_idx], y[train_idx])
            models.append(model)
        self._models = models

    def predict_proba(self, X):
        results = np.zeros(X.shape[0])
        for model in self._models:
            results += model.predict_proba(X)[:, 1]
        proba = (results/len(self._models)).reshape(-1, 1)
        return np.hstack((1-proba, proba))
