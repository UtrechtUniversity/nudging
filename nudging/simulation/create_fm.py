import numpy as np

from .base import BasePipe
from .utils import Bounds


class CreateFM(BasePipe):
    def __init__(self,
                 n_samples=np.array([500, 5000], dtype=int),
                 nudge_avg=np.array([0.05, 0.5]),
                 control_unique=np.array([0, 1.0]),
                 control_precision=np.array([0.2, 1.0])):
        self.n_samples = Bounds(n_samples, int_val=True)
        self.nudge_avg = Bounds(nudge_avg)
        self.control_unique = Bounds(control_unique)
        self.control_precision = Bounds(control_precision)

    def execute(self, data):
        corr_matrix, truth = data
        n_samples = self.n_samples.rand()
        nudge_avg = self.nudge_avg.rand()
        control_unique = self.control_unique.rand()
        control_precision = self.control_precision.rand()

        n_features = corr_matrix.shape[0]-2
        L = np.linalg.cholesky(corr_matrix)
        X = np.dot(L, np.random.randn(n_features+2, n_samples)).T
        control_intrinsic = X[:, -1]
        nudge_intrinsic = X[:, -2]
        X = X[:, :-2]

        true_outcome_control = (control_intrinsic*control_unique
                                + (1-control_unique)*nudge_intrinsic)
        true_outcome_control *= control_precision
        true_outcome_nudge = nudge_intrinsic + nudge_avg

        truth.update({
            "n_samples": n_samples,
            "nudge_avg": nudge_avg,
            "control_unique": control_unique,
            "control_precision": control_precision,
        })
        return (X, true_outcome_control, true_outcome_nudge, truth)
