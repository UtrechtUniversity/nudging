"""Base class for DataSet """
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseDataSet(ABC):
    """Base class for DataSet """
    covariates = []
    nudge_type = None
    nudge_domain = None
    compare = lambda a, b: None

    def __init__(self, file_path):
        self.filename = file_path
        self.raw_df = self._load(file_path)
        self.standard_df = self._preprocess(self.raw_df)

    @abstractmethod
    def _load(self, file_path):
        raise NotImplementedError

    def _preprocess(self, data_frame):
        # Remove unused columns
        used_columns = self.covariates + ["nudge", "outcome"]
        data_frame = data_frame.loc[:, used_columns]

        # Convert to a numeric type
        data_frame = data_frame.apply(pd.to_numeric)

        # Remove rows with NA in them
        data_frame = data_frame.dropna()

        return data_frame

    def write_raw(self, path):
        """Write raw data to csv file"""
        self.raw_df.to_csv(path, index=False)

    def write_interim(self, data_frame, path):
        """Write interim data (standard format) to csv file"""
        data_frame["nudge_type"] = self.nudge_type
        data_frame["nudge_domain"] = self.nudge_domain
        data_frame.to_csv(path, index=False)

    def get_success(self, data_frame):
        """Convert outcome to nudge success"""
        if self.compare not in [np.greater, np.less]:
            raise RuntimeError(
                f'Comparing outcome failed with {self.compare}, \
                self.compare should be np.greater or np.less')


        data_frame["success"] = self.compare(
            data_frame["outcome"], data_frame["control"]).astype(int)
        result = data_frame.drop(columns=["outcome", "control"])
        return result
