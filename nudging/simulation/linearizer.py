import numpy as np
from .base import BasePipe


def _transform_outcome(outcome, a, powers=np.array([1, 0.5, 0.1])):
    ret_outcome = np.zeros_like(outcome)
    a *= powers
    for i in range(len(a)):
        ret_outcome += (a[i]-powers[i]/2)*outcome**(i+1)
    return ret_outcome


class Linearizer(BasePipe):
    def __init__(self, linear=None):
        self.linear = linear

    def execute(self, X_true_outcome):
        X, true_outcome_control, true_outcome_nudge, truth = X_true_outcome
        if self.linear is None:
            linear = np.bool_(np.random.randint(2))
        else:
            linear = self.linear
        truth["linear"] = linear

        if linear:
            return X_true_outcome
        n_features = X.shape[1]

        a = np.random.rand(3)
        true_outcome_control = _transform_outcome(true_outcome_control, a)
        true_outcome_nudge = _transform_outcome(true_outcome_nudge, a)
        for i_col in range(n_features):
            a = np.random.rand(3)
            X[:, i_col] = _transform_outcome(X[:, i_col], a)

        return (X, true_outcome_control, true_outcome_nudge, truth)
