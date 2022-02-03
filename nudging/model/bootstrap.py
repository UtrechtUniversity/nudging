import numpy as np

from nudging.model.base import BaseModel
from nudging.dataset.base import bootstrap


class BootstrapModel(BaseModel):
    def __init__(self, model, n_bootstrap=100):
        super().__init__(model, predictors=model.predictors)
        self._boot_models = [self.model.clone() for _ in range(n_bootstrap)]

    def train(self, data):
        self.model.train(data)
        for m in self._boot_models:
            m.train(bootstrap(data))

    def predict_cate(self, data):
        single_model_cate = self.model.predict_cate(data)
        multi_res = np.array([m.predict_cate(data) for m in self._boot_models])
        return {
            "single_cate": single_model_cate,
            "multi_cate": multi_res,
        }
