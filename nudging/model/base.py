""" Model classes"""
from abc import ABC
import numpy as np


class BaseModel(ABC):
    """Base class for model"""

    def __init__(self, model, *args, **kwargs):
        self.model = model(*args, **kwargs)

    def fit(self, data, outcome):
        self.model.fit(data, outcome)

    def predict(self, data):
        return self.model.predict(data) 

    def predict_proba(self, data):
        return self.model.predict_proba(data) 
   
    def fit_nudge(self, X, outcome, nudge):
        X_new = np.hstack((X, nudge.reshape(-1, 1)))
        try:
            self.fit(X_new, outcome)
        except ValueError:
            y_new = outcome >= np.quantile(outcome, 0.5)
            if np.min(outcome) == np.quantile(outcome, 0.5):
                y_new = outcome > np.quantile(outcome, 0.5)
            self.model.fit(X_new, y_new)

    def predict_nudge(self, X, nudge):
        X_new = np.hstack((X, nudge.reshape(-1, 1)))
        return self.model.predict_proba(X_new)[:, 1]

    def predict_cate(self, X, return_avg=False):
        nudge_pred = self.predict_nudge(X, np.ones(X.shape[0]))
        control_pred = self.predict_nudge(X, np.zeros(X.shape[0]))
        cate = nudge_pred-control_pred
        if not return_avg:
            return cate
        return cate, (nudge_pred+control_pred)/2


class BaseRegressor(BaseModel):
    """Base class for Regressor model"""

    def fit_nudge(self, X, outcome, nudge):
        X_new = np.hstack((X, nudge.reshape(-1, 1)))
        self.model.fit(X_new, outcome)

    def predict_nudge(self, X, nudge):
        X_new = np.hstack((X, nudge.reshape(-1, 1)))
        return self.model.predict(X_new)


class BaseBiRegressor(BaseModel):
    """Base class for BiRegressor model"""
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.model_nudge = model(*args, **kwargs)
        self.model_control = model(*args, **kwargs)

    def fit_nudge(self, X, outcome, nudge):
        nudge_idx = np.where(nudge == 1)[0]
        control_idx = np.where(nudge == 0)[0]
        self.model_nudge.fit(X[nudge_idx], outcome[nudge_idx])
        self.model_control.fit(X[control_idx], outcome[control_idx])

    def predict_nudge(self, X, nudge):
        nudge_idx = np.where(nudge == 1)[0]
        control_idx = np.where(nudge == 0)[0]
        result = np.zeros(len(nudge))
        if len(nudge_idx):
            result[nudge_idx] = self.model_nudge.predict(X[nudge_idx])
        if len(control_idx):
            result[control_idx] = self.model_control.predict(X[control_idx])
        return result


class BaseXRegressor(BaseRegressor):
    """Base class for XRegressor model"""
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.base_model = BaseBiRegressor(model, *args, **kwargs)
        self.nudge_control_model = model(*args, **kwargs)
        self.control_nudge_model = model(*args, **kwargs)

    def fit_nudge(self, X, outcome, nudge):
        self.base_model.fit_nudge(X, outcome, nudge)
        nudge_idx = np.where(nudge == 1)[0]
        control_idx = np.where(nudge == 0)[0]
        imputed_treatment_nudge = self.base_model.predict_nudge(
            X[control_idx], 1-nudge[control_idx]) - outcome[control_idx]
        imputed_treatment_control = outcome[nudge_idx] - self.base_model.predict_nudge(
            X[nudge_idx], 1-nudge[nudge_idx])
        self.nudge_control_model.fit(X[nudge_idx], imputed_treatment_control)
        self.control_nudge_model.fit(X[control_idx], imputed_treatment_nudge)

    def predict_nudge(self, X, nudge):
        control_cate = self.control_nudge_model.predict(X)
        nudge_cate = self.nudge_control_model.predict(X)
        cate = 0.5*(control_cate + nudge_cate)
        base_pred = self.base_model.predict_nudge(X, np.zeros_like(nudge))
        return base_pred + cate*nudge

    def predict_cate(self, X, return_avg=False):
        control_cate = self.control_nudge_model.predict(X)
        nudge_cate = self.nudge_control_model.predict(X)
        cate = 0.5*(control_cate + nudge_cate)
        if not return_avg:
            return cate
        base_pred = self.base_model.predict_nudge(X, np.zeros(len(cate)))
        return cate, base_pred+0.5*cate