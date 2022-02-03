import numpy as np

from .base import BasePipe
from .utils import Bounds


class GenNudgeOutcome(BasePipe):
    def __init__(self, balance=np.array([0.2, 0.8])):
        self.balance = Bounds(balance)

    def execute(self, X_true_outcome):
        balance = self.balance.rand()
        X, true_outcome_control, true_outcome_nudge, truth = X_true_outcome
        n_samples = truth["n_samples"]
        n_treat = round(n_samples*balance)
        n_treat = min(max(2, n_treat), n_samples-2)
        nudge = np.zeros(n_samples, dtype=int)
        nudge[np.random.choice(n_samples, n_treat, replace=False)] = 1
        cate = true_outcome_nudge - true_outcome_control
        outcome = (true_outcome_control*(1-nudge)
                   + true_outcome_nudge*nudge)
        truth["balance"] = balance
        truth["cate"] = cate
        return (X, nudge, outcome, truth)
