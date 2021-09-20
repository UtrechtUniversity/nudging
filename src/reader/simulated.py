"""DataSet class for Pennycook et al 2020
(https://journals.sagepub.com/doi/10.1177/0956797620939054)"""
import numpy as np
import pandas as pd

from reader.base import BaseDataSet


class Simulated(BaseDataSet):
    """DataSet class for Pennycook et al 2020 """
    # Columns to keep as covariates
    covariates = ["age", "gender"]


    def _load(self, file_path):
        """ Read file and return data in dataframe
        Args:
            file_path (str): path of file
        Returns:
            pandas.DataFrame: raw data in dataframe
        """
        return pd.read_csv(file_path, encoding="iso-8859-1")

    def write_interim(self, data_frame, path):
        """Write interim data (standard format) to csv file"""
        data_frame["nudge_type"] = self.raw_df["nudge_type"]
        data_frame["nudge_domain"] = self.raw_df["nudge_domain"]
        data_frame.to_csv(path, index=False)