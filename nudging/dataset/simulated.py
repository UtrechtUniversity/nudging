"""DataSet class for simulated data"""
import pandas as pd

from nudging.dataset.base import BaseDataSet


class Simulated(BaseDataSet):
    """DataSet class for Pennycook et al 2020 """
    # Columns to keep as covariates
    covariates = ["age", "gender"]
    goal = "increase"

    def _load(self, file_path):
        """ Read file and return data in dataframe
        Args:
            file_path (str): path of file
        Returns:
            pandas.DataFrame: raw data in dataframe
        """
        return pd.read_csv(file_path, encoding="iso-8859-1")


    def write_interim(self, path):
        """Write interim data (standard format) to csv file"""
        self.standard_df["nudge_type"] = self.raw_df["nudge_type"]
        self.standard_df["nudge_domain"] = self.raw_df["nudge_domain"]

        super().write_interim(path)
