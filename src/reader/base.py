"""Base class for DataSet """
from abc import ABC, abstractmethod

import pandas as pd


class BaseDataSet(ABC):
    """Base class for DataSet """
    covariates = []
    nudge_type = None
    nudge_domain = None

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
