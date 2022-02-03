import numpy as np

from .base import BasePipe
from .utils import Bounds


class AddNoise(BasePipe):
    def __init__(self, noise_frac=np.array([0, 0.99])):
        self.noise_frac = Bounds(noise_frac)

    def execute(self, X_nudge_outcome):
        noise_frac = self.noise_frac.rand()
        X, nudge, outcome, truth = X_nudge_outcome
        n_samples = X.shape[0]
        outcome += (noise_frac/(1-noise_frac))*np.random.randn(n_samples)
        truth["noise_frac"] = noise_frac
        return (X, nudge, outcome, truth)
