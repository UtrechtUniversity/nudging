import numpy as np

from .base import BasePipe
from .utils import Bounds


def find_free_col(X, n_features_uncorrelated=None):
    feature_names = list(X.standard_df)
    if n_features_uncorrelated is None:
        np.random.shuffle(feature_names)
    for i, name in enumerate(feature_names):
        try:
            int(name)
            return name
        except ValueError:
            pass
        if (n_features_uncorrelated is not None
                and i >= n_features_uncorrelated):
            break
    raise ValueError("Cannot find free column in conversion process.")


def rescale(var, min_value, max_value):
    """ Rescale data to given range
    Args:
        var (series): data to be rescaled
        min_value (int): min value of new range
        max_value (int): max value of new range
    Returns:
        series: rescaled data
    """
    result = (((var - var.min())/(var.max() - var.min() + 1e-12)) *
              (max_value - min_value) + min_value)
    return result.astype(int)


class ConvertAge(BasePipe):
    def execute(self, X):
        truth = X.truth
        col = find_free_col(
            X, n_features_uncorrelated=truth["n_features_uncorrelated"])
        X.standard_df[col] = rescale(X.standard_df[col].values, 18, 80)
        X.standard_df.rename(columns={col: "age"}, inplace=True)
        if "n_rescale" in truth:
            truth["n_rescale"] += 1
        else:
            truth["n_rescale"] = 1
        return X


class ConvertGender(BasePipe):
    def execute(self, X):
        truth = X.truth
        col = find_free_col(
            X, n_features_uncorrelated=truth["n_features_uncorrelated"])
        X.standard_df[col] = rescale(X.standard_df[col].values, 0, 2)
        X.standard_df.rename(columns={col: "gender"}, inplace=True)
        if "n_rescale" in truth:
            truth["n_rescale"] += 1
        else:
            truth["n_rescale"] = 1
        return X


class Categorical(BasePipe):
    def __init__(self, n_rescale=np.array([0, 4], dtype=int),
                 n_layers=np.array([0, 5], dtype=int)):
        self.n_rescale = Bounds(n_rescale, int_val=True)
        self.n_layers = Bounds(n_layers, int_val=True)

    def execute(self, X):
        truth = X.truth
        n_rescale = self.n_rescale.rand()
        if "n_rescale" not in truth:
            truth["n_rescale"] = 0
        for _ in range(n_rescale):
            try:
                col = find_free_col(X)
            except ValueError:
                break
            X.standard_df[col] = rescale(X.standard_df[col].values, 0,
                                         self.n_layers.rand())
            X.standard_df.rename(columns={col: f"layer_{col}"}, inplace=True)
            truth["n_rescale"] += 1
        return X
