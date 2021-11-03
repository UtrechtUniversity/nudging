from .base import BaseRegressor
from sklearn.ensemble import RandomForestRegressor
from nudge.model.base import BaseBiRegressor, BaseXRegressor


class RFModel(BaseRegressor, RandomForestRegressor):
    pass


class RFBiModel(BaseBiRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(model=RandomForestRegressor, *args, **kwargs)


class RFXModel(BaseXRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(model=RandomForestRegressor, *args, **kwargs)
