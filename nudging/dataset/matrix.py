"""DataSet class for simlated matrix data"""
from pandas import DataFrame

from nudging.dataset.base import BaseDataSet


class MatrixData(BaseDataSet):
    """Class MatrixData"""
    @classmethod
    def from_data(cls, data, truth=None, names=None, **kwargs):
        """Initialize dataset from numpy arrays.

        Arguments
        ---------
        X: np.ndarray
            Feature matrix in numpy array format (NxM).
        outcome: np.ndarray
            Outcome for each of the samples (N).
        nudge: np.ndarray
            Whether each subject was nudged or not (1 or 0) (N).
        names: list[str]
            List of column names (M)

        Returns
        -------
        MatrixData:
            Initialized dataset.
        """
        X, nudge, outcome = data
        if truth is None:
            truth = {}
        truth.update(kwargs)
        standard_df = DataFrame(X)
        if names is not None:
            standard_df.set_axis(names, axis=1, inplace=True)
        else:
            standard_df.set_axis(
                [str(x) for x in list(standard_df)], axis=1, inplace=True)
        standard_df["outcome"] = outcome
        standard_df["nudge"] = nudge
        if "nudge_type" not in truth:
            truth["nudge_type"] = -1
        if "nudge_domain" not in truth:
            truth["nudge_domain"] = -1
        return cls(standard_df=standard_df, truth=truth)
